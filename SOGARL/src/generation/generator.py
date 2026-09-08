"""
Generation and policy-scoring interface for SOGARL.

Responsibilities:
    - Load Red/Blue SFT LoRA policies
    - Support shared-backbone and independent-backbone policy layouts
    - Load trained checkpoint adapters
    - Generate candidate responses
    - Compute response-token log probabilities
    - Provide model/tokenizer access

Does NOT:
    - Build prompts
    - Load scenarios
    - Evaluate responses
    - Calculate rewards
    - Select Top-K
    - Perform GRPO optimization


MODEL LOADING
=============

Shared-backbone mode
--------------------

    USE_SHARED_BACKBONE=True

        one base model
            ├── Red LoRA   (adapter name: "red")
            └── Blue LoRA  (adapter name: "blue")


Independent-backbone mode
-------------------------

    USE_SHARED_BACKBONE=False

        base model #1
            └── Red LoRA   (adapter name: "red")

        base model #2
            └── Blue LoRA  (adapter name: "blue")


IMPORTANT DESIGN RULE
=====================

The public Generator interface is identical in both modes.

Callers must NOT care whether the backbone is shared.

Adapter names are also identical in both modes:

    Red  -> "red"
    Blue -> "blue"

This is important because GRPO may temporarily activate reference
adapters. Generator._activate_adapter() must always be able to restore
the policy adapter afterward.


TOKENIZER RULE
==============

When possible, Generator prefers the tokenizer stored with the SFT
adapter itself.

This recreates the tokenizer/chat-template environment used during SFT
more faithfully than blindly reloading a tokenizer from the base model.

If the adapter does not contain tokenizer files, Generator falls back
to BASE_MODEL_NAME.


GENERATION RULE
===============

Autoregressive generation always runs temporarily in eval mode.

If the policy was previously in train mode:

    train
      ↓
    temporary eval
      ↓
    generate
      ↓
    restore train

GRPO teacher-forced log-probability scoring still runs in train mode
when require_grad=True.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from configs import config
from src.data.models import GenerationResult
from src.generation.chat_formatter import (
    format_chat_prompt,
)


# ============================================================================
# ROLE-SPECIFIC VIEW OVER A SHARED PEFT MODEL
# ============================================================================


class _RoleModelView(torch.nn.Module):
    """
    Role-specific view over one shared PEFT model.

    The underlying quantized base model is shared between Red and Blue.

    Each Generator receives its own lightweight view:

        Red Generator
            ↓
        _RoleModelView(adapter="red")
            ↓
        shared PeftModel

        Blue Generator
            ↓
        _RoleModelView(adapter="blue")
            ↓
        same shared PeftModel

    The view guarantees that forwards through this policy normally
    reactivate its own adapter.

    Exception:

        GRPO reference adapters use names beginning with
        "__sogarl_reference__".

    When one of those is intentionally active, forward() must not
    immediately replace it with the policy adapter. Reference-model
    wrappers explicitly manage that state themselves.
    """

    def __init__(
        self,
        model,
        adapter_name: str,
    ) -> None:

        super().__init__()

        self._shared_model = model
        self.adapter_name = adapter_name

    # ------------------------------------------------------------------
    # ADAPTER ACTIVATION
    # ------------------------------------------------------------------

    def _activate(self) -> None:

        if hasattr(
            self._shared_model,
            "set_adapter",
        ):
            self._shared_model.set_adapter(
                self.adapter_name
            )

        adapter_token = (
            f".{self.adapter_name}."
        )

        # --------------------------------------------------------------
        # Shared model may contain:
        #
        #   red
        #   blue
        #   red reference
        #   blue reference
        #
        # Only this view's policy adapter is trainable when activated.
        # --------------------------------------------------------------

        for (
            name,
            parameter,
        ) in self._shared_model.named_parameters():

            if adapter_token in name:

                parameter.requires_grad = True

            else:

                parameter.requires_grad = False

        trainable_count = sum(
            1
            for name, _ in (
                self._shared_model.named_parameters()
            )
            if adapter_token in name
        )

        if trainable_count == 0:

            raise RuntimeError(
                "No parameters were found for active "
                f"LoRA adapter '{self.adapter_name}'."
            )

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(
        self,
        *args,
        **kwargs,
    ):

        active = getattr(
            self._shared_model,
            "active_adapter",
            self.adapter_name,
        )

        if isinstance(
            active,
            (list, tuple),
        ):

            active_names = list(
                active
            )

        else:

            active_names = [
                active
            ]

        reference_active = any(
            isinstance(name, str)
            and name.startswith(
                "__sogarl_reference__"
            )
            for name in active_names
        )

        if not reference_active:

            self._activate()

        return self._shared_model(
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------

    def parameters(
        self,
        recurse: bool = True,
    ):

        self._activate()

        token = (
            f".{self.adapter_name}."
        )

        for (
            name,
            parameter,
        ) in self._shared_model.named_parameters(
            recurse=recurse
        ):

            if token in name:
                yield parameter

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):

        self._activate()

        token = (
            f".{self.adapter_name}."
        )

        for (
            name,
            parameter,
        ) in self._shared_model.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):

            if token in name:

                yield (
                    name,
                    parameter,
                )

    # ------------------------------------------------------------------
    # TRAIN / EVAL
    # ------------------------------------------------------------------

    def train(
        self,
        mode: bool = True,
    ):

        self._activate()

        self._shared_model.train(
            mode
        )

        self.training = mode

        return self

    def eval(
        self,
    ):

        self._activate()

        self._shared_model.eval()

        self.training = False

        return self

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def state_dict(
        self,
        *args,
        **kwargs,
    ):

        return self._shared_model.state_dict(
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ATTRIBUTE FORWARDING
    # ------------------------------------------------------------------

    def __getattr__(
        self,
        name,
    ):

        if name in {
            "_shared_model",
            "adapter_name",
        }:

            return super().__getattr__(
                name
            )

        return getattr(
            self._shared_model,
            name,
        )


# ============================================================================
# GENERATOR
# ============================================================================


class Generator:

    # ========================================================================
    # SHARED-BACKBONE PROCESS STATE
    # ========================================================================

    _shared_model = None
    _shared_tokenizer = None
    _shared_model_source = None
    _shared_device = None
    _shared_adapter_names = set()
    _shared_tokenizer_signature = None

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        role: str,
        model_name: str = "unknown",
        adapter_name: Optional[str] = None,
        max_input_tokens: int = (
            config.MAX_INPUT_TOKENS
        ),
        max_new_tokens: int = (
            config.MAX_NEW_TOKENS
        ),
        top_p: float = 0.95,
        do_sample: bool = True,
        generation_batch_size: int = 1,
        logger=None,
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer

        self.device = device

        self.role = role.lower()

        self.model_name = (
            model_name
        )

        self.adapter_name = (
            adapter_name
            if adapter_name is not None
            else self.role
        )

        self.max_input_tokens = (
            max_input_tokens
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        self.top_p = top_p

        self.do_sample = (
            do_sample
        )

        self.generation_batch_size = (
            generation_batch_size
        )

        self.logger = logger

        self._validate_config()

        self._prepare_tokenizer()

    # ========================================================================
    # SHARED STATE RESET
    # ========================================================================

    @classmethod
    def reset_shared_state(
        cls,
    ) -> None:
        """
        Forget Generator's process-level shared-backbone references.

        Useful for tests which intentionally reconstruct the model
        architecture inside the same Python process.

        This does NOT forcibly destroy models still referenced elsewhere.
        """

        cls._shared_model = None
        cls._shared_tokenizer = None

        cls._shared_model_source = None
        cls._shared_device = None

        cls._shared_adapter_names = set()

        cls._shared_tokenizer_signature = None

    # ========================================================================
    # MODEL LOADING FROM CONFIG
    # ========================================================================

    @classmethod
    def from_config(
        cls,
        role: str,
        logger=None,
        is_trainable: bool = True,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        adapter_path = (
            config.RED_ADAPTER_PATH
            if role == "red"
            else config.BLUE_ADAPTER_PATH
        )

        # ------------------------------------------------------------------
        # Explicit merged-model compatibility mode.
        # ------------------------------------------------------------------

        use_merged_model = bool(
            getattr(
                config,
                "USE_MERGED_MODEL",
                False,
            )
        )

        if use_merged_model:

            if is_trainable:

                raise RuntimeError(
                    "USE_MERGED_MODEL=True cannot be used with "
                    "is_trainable=True. A merged model no longer "
                    "contains a separately trainable LoRA adapter."
                )

            merged_model_path = (
                cls._find_merged_model(
                    role=role,
                    adapter_path=(
                        adapter_path
                    ),
                )
            )

            if merged_model_path is None:

                raise FileNotFoundError(
                    f"{role.capitalize()} merged-model loading "
                    "was explicitly enabled, but no valid merged "
                    "model was found. Check "
                    f"{role.upper()}_MERGED_MODEL_PATH."
                )

            if logger:

                logger.warning(
                    f"Loading {role.capitalize()} from "
                    "explicit MERGED MODEL configuration."
                )

                logger.warning(
                    "Merged-model mode is intended for "
                    "inference/evaluation compatibility."
                )

            return cls.from_merged_model(
                role=role,
                model_path=(
                    merged_model_path
                ),
                logger=logger,
            )

        # ------------------------------------------------------------------
        # Normal LoRA policy loading.
        # ------------------------------------------------------------------

        try:

            return cls.from_adapter(
                role=role,
                adapter_path=(
                    adapter_path
                ),
                logger=logger,
                is_trainable=(
                    is_trainable
                ),
            )

        except Exception as exc:

            use_shared = bool(
                getattr(
                    config,
                    "USE_SHARED_BACKBONE",
                    True,
                )
            )

            mode_name = (
                "shared-backbone"
                if use_shared
                else "independent-backbone"
            )

            raise RuntimeError(
                f"Failed to load the SOGARL "
                f"{role.capitalize()} policy using "
                f"{mode_name} LoRA mode. "
                "No merged model was loaded automatically. "
                "Check BASE_MODEL_NAME, adapter paths, tokenizer "
                "compatibility, quantization configuration, and "
                f"{role.upper()}_ADAPTER_PATH. "
                f"Original error: {exc}"
            ) from exc

    # ========================================================================
    # MERGED MODEL LOADING
    # ========================================================================

    @classmethod
    def from_merged_model(
        cls,
        role: str,
        model_path: str | Path,
        logger=None,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        model_path = (
            Path(model_path)
            .expanduser()
            .resolve()
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"{role.capitalize()} merged model "
                f"does not exist:\n"
                f"{model_path}"
            )

        if not model_path.is_dir():

            raise NotADirectoryError(
                f"{role.capitalize()} merged model "
                "path is not a directory:\n"
                f"{model_path}"
            )

        if not (
            model_path
            / "config.json"
        ).is_file():

            raise FileNotFoundError(
                "config.json was not found in "
                f"merged model:\n{model_path}"
            )

        device = (
            cls._resolve_device()
        )

        dtype = (
            cls._resolve_dtype(
                device
            )
        )

        if logger:

            logger.info(
                f"Loading {role.capitalize()} "
                "merged model"
            )

            logger.info(
                f"Model  : {model_path}"
            )

            logger.info(
                f"Device : {device}"
            )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(model_path),
                use_fast=True,
                local_files_only=True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        )

        model.to(
            device
        )

        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            role=role,
            adapter_name=None,
            model_name=str(
                model_path
            ),
            max_input_tokens=(
                config.MAX_INPUT_TOKENS
            ),
            max_new_tokens=(
                config.MAX_NEW_TOKENS
            ),
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            generation_batch_size=(
                config.GENERATION_BATCH_SIZE
            ),
            logger=logger,
        )

    # ========================================================================
    # FIND MERGED MODEL
    # ========================================================================

    @staticmethod
    def _find_merged_model(
        role: str,
        adapter_path: str | Path,
    ) -> Optional[Path]:

        role = role.lower()

        if role == "red":

            configured = getattr(
                config,
                "RED_MERGED_MODEL_PATH",
                None,
            )

        else:

            configured = getattr(
                config,
                "BLUE_MERGED_MODEL_PATH",
                None,
            )

        if configured:

            configured_path = (
                Path(configured)
                .expanduser()
                .resolve()
            )

            if (
                configured_path.is_dir()
                and (
                    configured_path
                    / "config.json"
                ).is_file()
            ):

                return (
                    configured_path
                )

        adapter_path = (
            Path(adapter_path)
            .expanduser()
            .resolve()
        )

        sibling_path = (
            adapter_path.parent
            / "merged_model"
        )

        if (
            sibling_path.is_dir()
            and (
                sibling_path
                / "config.json"
            ).is_file()
        ):

            return sibling_path

        return None

    # ========================================================================
    # ADAPTER TOKENIZER HELPERS
    # ========================================================================

    @staticmethod
    def _adapter_has_tokenizer(
        adapter_path: Path,
    ) -> bool:

        candidates = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        )

        return any(
            (
                adapter_path
                / filename
            ).is_file()
            for filename in candidates
        )

    @classmethod
    def _load_policy_tokenizer(
        cls,
        adapter_path: Path,
        model_source: str,
        local_files_only: bool,
        logger=None,
    ):
        """
        Prefer the tokenizer saved with the SFT adapter.

        SFT commonly saves:

            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

        Loading that tokenizer reproduces the SFT serialization more
        faithfully than assuming the base-model tokenizer is identical.
        """

        if cls._adapter_has_tokenizer(
            adapter_path
        ):

            if logger:

                logger.info(
                    "Tokenizer source: SFT adapter directory"
                )

                logger.info(
                    f"Tokenizer path : {adapter_path}"
                )

            try:

                return (
                    AutoTokenizer
                    .from_pretrained(
                        str(adapter_path),
                        use_fast=True,
                        local_files_only=True,
                    )
                )

            except Exception as exc:

                if logger:

                    logger.warning(
                        "Adapter-local tokenizer exists but "
                        "could not be loaded. Falling back to "
                        "base-model tokenizer."
                    )

                    logger.warning(
                        f"Tokenizer error: {exc}"
                    )

        if logger:

            logger.info(
                "Tokenizer source: base model"
            )

        return (
            AutoTokenizer
            .from_pretrained(
                model_source,
                use_fast=True,
                local_files_only=(
                    local_files_only
                ),
            )
        )

    @staticmethod
    def _tokenizer_signature(
        tokenizer,
    ) -> Tuple[Any, ...]:
        """
        Signature used to catch incompatible tokenizers in shared mode.

        We intentionally compare behaviorally important tokenizer fields,
        not path strings, because equivalent tokenizers may come from
        different directories.
        """

        chat_template = getattr(
            tokenizer,
            "chat_template",
            None,
        )

        template_hash = None

        if chat_template is not None:

            template_hash = (
                hashlib.sha256(
                    str(
                        chat_template
                    ).encode(
                        "utf-8"
                    )
                )
                .hexdigest()
            )

        return (
            tokenizer.__class__.__name__,
            getattr(
                tokenizer,
                "vocab_size",
                None,
            ),
            getattr(
                tokenizer,
                "bos_token_id",
                None,
            ),
            getattr(
                tokenizer,
                "eos_token_id",
                None,
            ),
            getattr(
                tokenizer,
                "pad_token_id",
                None,
            ),
            template_hash,
        )

    # ========================================================================
    # ADAPTER BASE-MODEL DIAGNOSTICS
    # ========================================================================

    @staticmethod
    def _read_adapter_base_model(
        adapter_path: Path,
    ) -> Optional[str]:

        config_path = (
            adapter_path
            / "adapter_config.json"
        )

        if not config_path.is_file():

            return None

        try:

            payload = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            return None

        base_name = payload.get(
            "base_model_name_or_path"
        )

        if base_name is None:

            return None

        return str(
            base_name
        )

    @classmethod
    def _log_adapter_compatibility(
        cls,
        adapter_path: Path,
        model_source: str,
        logger=None,
    ) -> None:

        expected_base = (
            cls._read_adapter_base_model(
                adapter_path
            )
        )

        if expected_base is None:

            if logger:

                logger.info(
                    "Adapter base-model metadata: unavailable"
                )

            return

        if logger:

            logger.info(
                "Adapter trained from : "
                f"{expected_base}"
            )

            logger.info(
                "Runtime base model   : "
                f"{model_source}"
            )

        # --------------------------------------------------------------
        # Do NOT hard-fail on string mismatch.
        #
        # Example:
        #
        #   unsloth/Meta-Llama-3.1-8B-Instruct
        #
        # and
        #
        #   unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit
        #
        # may still represent intentionally compatible model artifacts.
        # --------------------------------------------------------------

        if (
            expected_base.lower()
            != model_source.lower()
            and logger
        ):

            logger.warning(
                "Adapter base-model path differs from runtime "
                "BASE_MODEL_NAME. This is not automatically an "
                "error, but it should be verified if generation "
                "quality is abnormal."
            )

    # ========================================================================
    # ADAPTER LOADING
    # ========================================================================

    @classmethod
    def from_adapter(
        cls,
        role: str,
        adapter_path: str | Path,
        logger=None,
        is_trainable: bool = True,
        reload_adapter: bool = False,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        base_model_name = (
            config.BASE_MODEL_NAME
        )

        adapter_path = (
            Path(adapter_path)
            .expanduser()
            .resolve()
        )

        if not adapter_path.exists():

            raise FileNotFoundError(
                f"{role.capitalize()} LoRA adapter "
                f"does not exist:\n"
                f"{adapter_path}"
            )

        if not adapter_path.is_dir():

            raise NotADirectoryError(
                f"{role.capitalize()} LoRA adapter path "
                f"is not a directory:\n"
                f"{adapter_path}"
            )

        adapter_config_path = (
            adapter_path
            / "adapter_config.json"
        )

        if not adapter_config_path.is_file():

            raise FileNotFoundError(
                "adapter_config.json was not found in "
                f"{role.capitalize()} adapter:\n"
                f"{adapter_path}"
            )

        device = (
            cls._resolve_device()
        )

        dtype = (
            cls._resolve_dtype(
                device
            )
        )

        local_base_path = (
            cls._resolve_local_base_model(
                base_model_name
            )
        )

        if local_base_path is not None:

            model_source = str(
                local_base_path
            )

            local_files_only = True

        else:

            model_source = str(
                base_model_name
            )

            local_files_only = False

        use_shared_backbone = bool(
            getattr(
                config,
                "USE_SHARED_BACKBONE",
                True,
            )
        )

        if logger:

            logger.info(
                "================================================"
            )

            logger.info(
                f"LOADING {role.upper()} POLICY"
            )

            logger.info(
                f"Mode       : "
                f"{'SHARED' if use_shared_backbone else 'INDEPENDENT'}"
            )

            logger.info(
                f"Base model : {model_source}"
            )

            logger.info(
                f"Adapter    : {adapter_path}"
            )

            logger.info(
                f"Adapter ID : {role}"
            )

            logger.info(
                f"Device     : {device}"
            )

            logger.info(
                f"Trainable  : {is_trainable}"
            )

            logger.info(
                "================================================"
            )

        cls._log_adapter_compatibility(
            adapter_path=(
                adapter_path
            ),
            model_source=(
                model_source
            ),
            logger=logger,
        )

        try:

            from peft import (
                PeftModel,
                prepare_model_for_kbit_training,
            )

        except ImportError as exc:

            raise ImportError(
                "PEFT is required to load "
                "SOGARL LoRA adapters."
            ) from exc

        # ==================================================================
        # INDEPENDENT POLICY MODE
        # ==================================================================

        if not use_shared_backbone:

            # --------------------------------------------------------------
            # IMPORTANT:
            #
            # Independent policies get independent tokenizers too.
            #
            # This eliminates all Red/Blue tokenizer state coupling and
            # makes debugging SFT reconstruction considerably cleaner.
            # --------------------------------------------------------------

            tokenizer = (
                cls._load_policy_tokenizer(
                    adapter_path=(
                        adapter_path
                    ),
                    model_source=(
                        model_source
                    ),
                    local_files_only=(
                        local_files_only
                    ),
                    logger=logger,
                )
            )

            model_load_kwargs = {
                "local_files_only": (
                    local_files_only
                ),
            }

            if config.USE_4BIT_MODEL:

                if device.type != "cuda":

                    raise RuntimeError(
                        "USE_4BIT_MODEL=True requires CUDA."
                    )

                try:

                    import bitsandbytes  # noqa: F401

                except ImportError as exc:

                    raise ImportError(
                        "bitsandbytes is required when "
                        "USE_4BIT_MODEL=True."
                    ) from exc

                quantization_config = (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=(
                            dtype
                        ),
                    )
                )

                model_load_kwargs.update(
                    {
                        "quantization_config": (
                            quantization_config
                        ),
                        "device_map": {
                            "": (
                                device.index
                                if device.index
                                is not None
                                else 0
                            )
                        },
                    }
                )

            else:

                model_load_kwargs[
                    "torch_dtype"
                ] = dtype

            if logger:

                logger.info(
                    f"Loading independent "
                    f"{role.capitalize()} base model..."
                )

                if config.USE_4BIT_MODEL:

                    logger.info(
                        "Quantization: 4-bit NF4"
                    )

            base_model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_source,
                    **model_load_kwargs,
                )
            )

            # --------------------------------------------------------------
            # Prepare quantized model for LoRA training only when needed.
            # --------------------------------------------------------------

            if (
                config.USE_4BIT_MODEL
                and is_trainable
            ):

                base_model = (
                    prepare_model_for_kbit_training(
                        base_model,
                        use_gradient_checkpointing=(
                            config.USE_GRADIENT_CHECKPOINTING
                        ),
                    )
                )

            # --------------------------------------------------------------
            # CRITICAL:
            #
            # Explicitly name independent adapters "red" / "blue".
            #
            # Previously independent mode allowed PEFT to name the adapter
            # "default", while Generator.adapter_name remained "red"/"blue".
            #
            # That becomes unsafe once GRPO reference adapters are added,
            # because Generator must reliably reactivate the policy adapter.
            # --------------------------------------------------------------

            model = (
                PeftModel
                .from_pretrained(
                    base_model,
                    str(adapter_path),
                    adapter_name=role,
                    is_trainable=(
                        is_trainable
                    ),
                )
            )

            model.set_adapter(
                role
            )

            if is_trainable:

                model.train()

                trainable_parameters = [
                    name
                    for (
                        name,
                        parameter,
                    ) in model.named_parameters()
                    if parameter.requires_grad
                ]

                if not trainable_parameters:

                    raise RuntimeError(
                        "PEFT loaded the independent "
                        f"{role} adapter but exposed no "
                        "trainable parameters."
                    )

                if logger:

                    logger.info(
                        f"{role.capitalize()} trainable "
                        f"parameter tensors: "
                        f"{len(trainable_parameters)}"
                    )

            else:

                for parameter in (
                    model.parameters()
                ):

                    parameter.requires_grad = (
                        False
                    )

                model.eval()

            if logger:

                logger.info(
                    f"Independent {role.capitalize()} "
                    "policy loaded successfully."
                )

            return cls(
                model=model,
                tokenizer=tokenizer,
                device=device,
                role=role,
                adapter_name=role,
                model_name=model_source,
                max_input_tokens=(
                    config.MAX_INPUT_TOKENS
                ),
                max_new_tokens=(
                    config.MAX_NEW_TOKENS
                ),
                top_p=config.TOP_P,
                do_sample=config.DO_SAMPLE,
                generation_batch_size=(
                    config.GENERATION_BATCH_SIZE
                ),
                logger=logger,
            )

        # ==================================================================
        # SHARED-BACKBONE MODE
        # ==================================================================

        first_load = (
            cls._shared_model is None
            or cls._shared_model_source
            != model_source
            or cls._shared_device
            != device
        )

        # ------------------------------------------------------------------
        # Shared tokenizer handling.
        #
        # First policy establishes the tokenizer.
        # Second policy's saved tokenizer is checked for compatibility.
        # ------------------------------------------------------------------

        candidate_tokenizer = (
            cls._load_policy_tokenizer(
                adapter_path=(
                    adapter_path
                ),
                model_source=(
                    model_source
                ),
                local_files_only=(
                    local_files_only
                ),
                logger=logger,
            )
        )

        candidate_signature = (
            cls._tokenizer_signature(
                candidate_tokenizer
            )
        )

        if (
            first_load
            or cls._shared_tokenizer
            is None
        ):

            tokenizer = (
                candidate_tokenizer
            )

            cls._shared_tokenizer = (
                tokenizer
            )

            cls._shared_tokenizer_signature = (
                candidate_signature
            )

        else:

            if (
                cls._shared_tokenizer_signature
                != candidate_signature
            ):

                raise RuntimeError(
                    "Red and Blue tokenizers are not compatible "
                    "enough to safely use USE_SHARED_BACKBONE=True. "
                    "Their vocab/special-token/chat-template "
                    "signatures differ. Use "
                    "USE_SHARED_BACKBONE=False or repair the "
                    "SFT tokenizer artifacts."
                )

            tokenizer = (
                cls._shared_tokenizer
            )

        # ------------------------------------------------------------------
        # First shared-model load.
        # ------------------------------------------------------------------

        if first_load:

            model_load_kwargs = {
                "local_files_only": (
                    local_files_only
                ),
            }

            if config.USE_4BIT_MODEL:

                if device.type != "cuda":

                    raise RuntimeError(
                        "USE_4BIT_MODEL=True requires CUDA."
                    )

                try:

                    import bitsandbytes  # noqa: F401

                except ImportError as exc:

                    raise ImportError(
                        "bitsandbytes is required when "
                        "USE_4BIT_MODEL=True."
                    ) from exc

                quantization_config = (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=(
                            dtype
                        ),
                    )
                )

                model_load_kwargs.update(
                    {
                        "quantization_config": (
                            quantization_config
                        ),
                        "device_map": {
                            "": (
                                device.index
                                if device.index
                                is not None
                                else 0
                            )
                        },
                    }
                )

            else:

                model_load_kwargs[
                    "torch_dtype"
                ] = dtype

            if logger:

                logger.info(
                    "Loading ONE shared base model."
                )

                if config.USE_4BIT_MODEL:

                    logger.info(
                        "Quantization: 4-bit NF4"
                    )

            base_model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_source,
                    **model_load_kwargs,
                )
            )

            if (
                config.USE_4BIT_MODEL
                and is_trainable
            ):

                base_model = (
                    prepare_model_for_kbit_training(
                        base_model,
                        use_gradient_checkpointing=(
                            config.USE_GRADIENT_CHECKPOINTING
                        ),
                    )
                )

            model = (
                PeftModel
                .from_pretrained(
                    base_model,
                    str(adapter_path),
                    adapter_name=role,
                    is_trainable=(
                        is_trainable
                    ),
                )
            )

            model.set_adapter(
                role
            )

            cls._shared_model = (
                model
            )

            cls._shared_model_source = (
                model_source
            )

            cls._shared_device = (
                device
            )

            cls._shared_adapter_names = {
                role
            }

            if logger:

                logger.info(
                    "Shared backbone initialized with "
                    f"{role.capitalize()} adapter."
                )

        # ------------------------------------------------------------------
        # Shared base already exists.
        # ------------------------------------------------------------------

        else:

            model = (
                cls._shared_model
            )

            adapter_exists = (
                role
                in cls._shared_adapter_names
            )

            if (
                adapter_exists
                and reload_adapter
            ):

                if logger:

                    logger.info(
                        f"Reloading existing shared "
                        f"{role.capitalize()} adapter."
                    )

                # ----------------------------------------------------------
                # Do not silently keep stale checkpoint weights.
                # ----------------------------------------------------------

                delete_adapter = getattr(
                    model,
                    "delete_adapter",
                    None,
                )

                if delete_adapter is None:

                    raise RuntimeError(
                        "PEFT model does not support adapter deletion, "
                        "so an already-loaded shared adapter cannot be "
                        "replaced safely in this process. Call "
                        "Generator.reset_shared_state() before loading "
                        "the checkpoint."
                    )

                active = getattr(
                    model,
                    "active_adapter",
                    None,
                )

                if active == role:

                    # Switch to another known adapter before deletion
                    # when possible.
                    alternatives = [
                        name
                        for name in (
                            cls._shared_adapter_names
                        )
                        if name != role
                    ]

                    if alternatives:

                        model.set_adapter(
                            alternatives[0]
                        )

                model.delete_adapter(
                    role
                )

                cls._shared_adapter_names.discard(
                    role
                )

                adapter_exists = False

            if not adapter_exists:

                if logger:

                    logger.info(
                        f"Reusing shared base model for "
                        f"{role.capitalize()}."
                    )

                    logger.info(
                        "Loading LoRA adapter only."
                    )

                model.load_adapter(
                    str(adapter_path),
                    adapter_name=role,
                    is_trainable=(
                        is_trainable
                    ),
                )

                cls._shared_adapter_names.add(
                    role
                )

            elif logger:

                logger.info(
                    f"{role.capitalize()} adapter already "
                    "exists on shared backbone."
                )

        model.set_adapter(
            role
        )

        model_view = (
            _RoleModelView(
                model=model,
                adapter_name=role,
            )
        )

        if is_trainable:

            model_view.train(
                True
            )

        else:

            model_view.eval()

            for parameter in (
                model_view.parameters()
            ):

                parameter.requires_grad = (
                    False
                )

        return cls(
            model=model_view,
            tokenizer=tokenizer,
            device=device,
            role=role,
            adapter_name=role,
            model_name=model_source,
            max_input_tokens=(
                config.MAX_INPUT_TOKENS
            ),
            max_new_tokens=(
                config.MAX_NEW_TOKENS
            ),
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            generation_batch_size=(
                config.GENERATION_BATCH_SIZE
            ),
            logger=logger,
        )

    # ========================================================================
    # LOCAL BASE MODEL RESOLUTION
    # ========================================================================

    @staticmethod
    def _resolve_local_base_model(
        base_model_name: str | Path,
    ) -> Optional[Path]:

        path = (
            Path(
                str(base_model_name)
            )
            .expanduser()
        )

        if not path.exists():

            return None

        path = path.resolve()

        if not path.is_dir():

            return None

        if not (
            path
            / "config.json"
        ).is_file():

            return None

        return path

    # ========================================================================
    # CHECKPOINT LOADING
    # ========================================================================

    @classmethod
    def from_checkpoint(
        cls,
        role: str,
        checkpoint_path: str | Path,
        logger=None,
        is_trainable: bool = False,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        checkpoint_path = (
            Path(checkpoint_path)
            .expanduser()
            .resolve()
        )

        adapter_path = (
            checkpoint_path
            / (
                "red_adapter"
                if role == "red"
                else "blue_adapter"
            )
        )

        if not adapter_path.exists():

            raise FileNotFoundError(
                f"{role.capitalize()} checkpoint "
                "adapter does not exist:\n"
                f"{adapter_path}"
            )

        if logger:

            logger.info(
                f"Loading {role.capitalize()} "
                "policy from checkpoint"
            )

            logger.info(
                f"Checkpoint : {checkpoint_path}"
            )

        return cls.from_adapter(
            role=role,
            adapter_path=adapter_path,
            logger=logger,
            is_trainable=(
                is_trainable
            ),
            reload_adapter=True,
        )

    # ========================================================================
    # DEVICE / DTYPE
    # ========================================================================

    @staticmethod
    def _resolve_device() -> torch.device:

        requested = str(
            config.DEVICE
        ).lower()

        if requested.startswith(
            "cuda"
        ):

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "CUDA was requested but "
                    "is not available."
                )

            return torch.device(
                requested
            )

        if requested == "cpu":

            return torch.device(
                "cpu"
            )

        return torch.device(
            requested
        )

    @staticmethod
    def _resolve_dtype(
        device: torch.device,
    ) -> torch.dtype:

        if device.type != "cuda":

            return torch.float32

        if config.USE_BF16:

            if not (
                torch.cuda
                .is_bf16_supported()
            ):

                raise RuntimeError(
                    "USE_BF16=True but this CUDA "
                    "device does not support bfloat16."
                )

            return torch.bfloat16

        if config.USE_FP16:

            return torch.float16

        return torch.float32

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_role(
        role: str,
    ) -> str:

        role = role.lower()

        if role not in {
            "red",
            "blue",
        }:

            raise ValueError(
                "role must be either "
                "'red' or 'blue'."
            )

        return role

    def _validate_config(
        self,
    ) -> None:

        if self.role not in {
            "red",
            "blue",
        }:

            raise ValueError(
                "role must be either "
                "'red' or 'blue'."
            )

        if (
            self.max_input_tokens
            <= 0
        ):

            raise ValueError(
                "max_input_tokens must be "
                "greater than 0."
            )

        if (
            self.max_new_tokens
            <= 0
        ):

            raise ValueError(
                "max_new_tokens must be "
                "greater than 0."
            )

        if not (
            0.0
            < self.top_p
            <= 1.0
        ):

            raise ValueError(
                "top_p must be in "
                "the range (0, 1]."
            )

        if (
            self.generation_batch_size
            <= 0
        ):

            raise ValueError(
                "generation_batch_size must "
                "be greater than 0."
            )

    def _prepare_tokenizer(
        self,
    ) -> None:

        if (
            self.tokenizer.pad_token_id
            is None
        ):

            if (
                self.tokenizer.eos_token_id
                is None
            ):

                raise ValueError(
                    "Tokenizer has neither "
                    "pad_token_id nor "
                    "eos_token_id."
                )

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.tokenizer.padding_side = (
            "left"
        )

    # ========================================================================
    # INTERNAL MODEL HELPERS
    # ========================================================================

    def _underlying_model(
        self,
    ):
        """
        Return actual PEFT/causal-LM model.

        Shared mode:
            _RoleModelView -> shared PeftModel

        Independent mode:
            PeftModel -> itself

        Merged mode:
            AutoModelForCausalLM -> itself
        """

        return getattr(
            self.model,
            "_shared_model",
            self.model,
        )

    def _model_training_state(
        self,
    ) -> bool:

        underlying = (
            self._underlying_model()
        )

        return bool(
            underlying.training
        )

    # ========================================================================
    # ADAPTER ACTIVATION
    # ========================================================================

    def _activate_adapter(
        self,
    ) -> None:
        """
        Restore this Generator's policy adapter.

        This works in BOTH modes.

        Shared:
            _RoleModelView._activate()

        Independent:
            direct PeftModel.set_adapter("red"/"blue")

        Merged:
            no adapter -> no-op
        """

        model = self.model

        if hasattr(
            model,
            "_activate",
        ):

            model._activate()

            return

        if hasattr(
            model,
            "set_adapter",
        ):

            try:

                model.set_adapter(
                    self.adapter_name
                )

                return

            except Exception:

                # Non-PEFT merged models do not normally expose
                # set_adapter(), but leave this defensive fallback.
                pass

        shared = getattr(
            model,
            "_shared_model",
            None,
        )

        if (
            shared is not None
            and hasattr(
                shared,
                "set_adapter",
            )
        ):

            shared.set_adapter(
                self.adapter_name
            )

    # ========================================================================
    # TEMPORARY GENERATION MODE
    # ========================================================================

    def _enter_generation_mode(
        self,
    ) -> bool:
        """
        Activate this policy and enter eval mode.

        Returns the previous underlying training state so the caller can
        restore it afterward.
        """

        self._activate_adapter()

        was_training = (
            self._model_training_state()
        )

        self.model.eval()

        # --------------------------------------------------------------
        # Generation should use cache when possible.
        #
        # Some training paths previously set config.use_cache=False.
        # Restore it for autoregressive inference.
        # --------------------------------------------------------------

        underlying = (
            self._underlying_model()
        )

        for model_object in (
            underlying,
            getattr(
                underlying,
                "base_model",
                None,
            ),
        ):

            if model_object is None:

                continue

            model_config = getattr(
                model_object,
                "config",
                None,
            )

            if model_config is not None:

                model_config.use_cache = (
                    True
                )

        return was_training

    def _restore_after_generation(
        self,
        was_training: bool,
    ) -> None:

        self._activate_adapter()

        if was_training:

            self.model.train()

        else:

            self.model.eval()

    # ========================================================================
    # GENERATION
    # ========================================================================

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.8,
        generation_indices: Optional[
            List[int]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        turn: int = 0,
        generation_type: str = "",
    ) -> List[GenerationResult]:

        if not prompts:

            return []

        if (
            self.do_sample
            and temperature <= 0
        ):

            raise ValueError(
                "temperature must be greater "
                "than 0 when sampling is enabled."
            )

        if generation_indices is None:

            generation_indices = list(
                range(
                    len(prompts)
                )
            )

        if (
            len(generation_indices)
            != len(prompts)
        ):

            raise ValueError(
                "generation_indices must have "
                "the same length as prompts."
            )

        results: List[
            GenerationResult
        ] = []

        for start in range(
            0,
            len(prompts),
            self.generation_batch_size,
        ):

            end = min(
                start
                + self.generation_batch_size,
                len(prompts),
            )

            batch_results = (
                self._generate_batch(
                    prompts=(
                        prompts[
                            start:end
                        ]
                    ),
                    temperature=(
                        temperature
                    ),
                    generation_indices=(
                        generation_indices[
                            start:end
                        ]
                    ),
                    metadata=(
                        metadata
                    ),
                    turn=turn,
                    generation_type=(
                        generation_type
                    ),
                )
            )

            results.extend(
                batch_results
            )

        return results

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        generation_indices: List[int],
        metadata: Optional[
            Dict[str, Any]
        ],
        turn: int,
        generation_type: str,
    ) -> List[GenerationResult]:

        was_training = (
            self._enter_generation_mode()
        )

        try:

            formatted_prompts = [
                format_chat_prompt(
                    self.tokenizer,
                    prompt,
                )
                for prompt in prompts
            ]

            inputs = (
                self.tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=(
                        self.max_input_tokens
                    ),
                    return_attention_mask=True,
                    add_special_tokens=False,
                )
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for (
                    key,
                    value,
                ) in inputs.items()
            }

            outputs = (
                self.model.generate(
                    **inputs,
                    **self._generation_kwargs(
                        temperature
                    ),
                )
            )

            # --------------------------------------------------------------
            # Because inputs are left padded, every generated sequence has
            # the same tensor input width. Generation appends new tokens
            # after that width.
            # --------------------------------------------------------------

            input_width = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            results: List[
                GenerationResult
            ] = []

            for (
                index,
                output,
            ) in enumerate(
                outputs
            ):

                generated_tokens = (
                    output[
                        input_width:
                    ]
                )

                response = (
                    self.tokenizer
                    .decode(
                        generated_tokens,
                        skip_special_tokens=True,
                    )
                    .strip()
                )

                result_metadata = dict(
                    metadata or {}
                )

                result_metadata.update(
                    {
                        "role": (
                            self.role
                        ),
                        "adapter_name": (
                            self.adapter_name
                        ),
                        "prompt_tokens": int(
                            inputs[
                                "attention_mask"
                            ][index]
                            .sum()
                            .item()
                        ),
                        "response_tokens": int(
                            generated_tokens
                            .shape[-1]
                        ),
                    }
                )

                results.append(
                    GenerationResult(
                        response=(
                            response
                        ),
                        prompt=(
                            prompts[index]
                        ),
                        model_name=(
                            self.model_name
                        ),
                        temperature=(
                            temperature
                        ),
                        generation_index=(
                            generation_indices[
                                index
                            ]
                        ),
                        turn=turn,
                        generation_type=(
                            generation_type
                        ),
                        metadata=(
                            result_metadata
                        ),
                    )
                )

            return results

        finally:

            self._restore_after_generation(
                was_training
            )

    # ========================================================================
    # GENERATION PARAMETERS
    # ========================================================================

    def _generation_kwargs(
        self,
        temperature: float,
    ) -> Dict[str, Any]:

        kwargs: Dict[
            str,
            Any,
        ] = {
            "max_new_tokens": (
                self.max_new_tokens
            ),
            "do_sample": (
                self.do_sample
            ),
            "pad_token_id": (
                self.tokenizer
                .pad_token_id
            ),
            "use_cache": True,
        }

        if (
            self.tokenizer
            .eos_token_id
            is not None
        ):

            kwargs[
                "eos_token_id"
            ] = (
                self.tokenizer
                .eos_token_id
            )

        if self.do_sample:

            kwargs.update(
                {
                    "temperature": (
                        temperature
                    ),
                    "top_p": (
                        self.top_p
                    ),
                }
            )

        return kwargs

    # ========================================================================
    # RESPONSE LOG PROBABILITIES
    # ========================================================================

    def compute_log_probs(
        self,
        prompts: List[str],
        responses: List[str],
        require_grad: bool = True,
    ) -> List[torch.Tensor]:

        if (
            len(prompts)
            != len(responses)
        ):

            raise ValueError(
                "prompts and responses must "
                "have the same length."
            )

        if not prompts:

            return []

        # --------------------------------------------------------------
        # Differentiable policy scoring.
        # --------------------------------------------------------------

        if require_grad:

            self._prepare_training_forward()

            return [
                self._compute_single_log_probs(
                    prompt=prompt,
                    response=response,
                )
                for (
                    prompt,
                    response,
                ) in zip(
                    prompts,
                    responses,
                )
            ]

        # --------------------------------------------------------------
        # Frozen/reference scoring.
        # --------------------------------------------------------------

        was_training = (
            self._model_training_state()
        )

        self._activate_adapter()

        self.model.eval()

        try:

            with torch.no_grad():

                return [
                    self._compute_single_log_probs(
                        prompt=prompt,
                        response=response,
                    )
                    for (
                        prompt,
                        response,
                    ) in zip(
                        prompts,
                        responses,
                    )
                ]

        finally:

            self._activate_adapter()

            if was_training:

                self.model.train()

            else:

                self.model.eval()

    # ========================================================================
    # TRAINING FORWARD PREPARATION
    # ========================================================================

    def _prepare_training_forward(
        self,
    ) -> None:
        """
        Prepare this policy for differentiable GRPO scoring.

        This does not alter:

            rewards
            advantages
            objective
            candidate count
            KL formulation
            token log-probability semantics

        It only establishes correct execution state.
        """

        self._activate_adapter()

        self.model.train()

        underlying_model = (
            self._underlying_model()
        )

        if (
            config.USE_GRADIENT_CHECKPOINTING
        ):

            enable_checkpointing = getattr(
                underlying_model,
                "gradient_checkpointing_enable",
                None,
            )

            if enable_checkpointing is not None:

                try:

                    enable_checkpointing(
                        gradient_checkpointing_kwargs={
                            "use_reentrant": False,
                        }
                    )

                except TypeError:

                    enable_checkpointing()

        # --------------------------------------------------------------
        # Teacher-forced scoring does not need KV caching.
        # --------------------------------------------------------------

        for model_object in (
            underlying_model,
            getattr(
                underlying_model,
                "base_model",
                None,
            ),
        ):

            if model_object is None:

                continue

            model_config = getattr(
                model_object,
                "config",
                None,
            )

            if model_config is not None:

                model_config.use_cache = (
                    False
                )

    # ========================================================================
    # SINGLE RESPONSE LOG PROBABILITY
    # ========================================================================

    def _compute_single_log_probs(
        self,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """
        Return response-token log probabilities.

        The first returned value is:

            log P(response[0] | prompt)

        Prompt-token probabilities are excluded.

        Prompt formatting is identical to generation.
        """

        self._activate_adapter()

        formatted_prompt = (
            format_chat_prompt(
                self.tokenizer,
                prompt,
            )
        )

        prompt_tokens = (
            self.tokenizer(
                formatted_prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self.max_input_tokens
                ),
            )
        )

        response_tokens = (
            self.tokenizer(
                response,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self.max_new_tokens
                ),
            )
        )

        prompt_ids = (
            prompt_tokens[
                "input_ids"
            ][0]
        )

        response_ids = (
            response_tokens[
                "input_ids"
            ][0]
        )

        if (
            response_ids.numel()
            == 0
        ):

            return torch.empty(
                0,
                device=self.device,
                dtype=torch.float32,
            )

        prompt_length = int(
            prompt_ids.numel()
        )

        response_length = int(
            response_ids.numel()
        )

        # --------------------------------------------------------------
        # Generation conditions on a prompt truncated to
        # max_input_tokens.
        #
        # GRPO must score against that same context.
        #
        # Only truncate further if the underlying model's actual context
        # window makes prompt + response impossible.
        # --------------------------------------------------------------

        underlying = (
            self._underlying_model()
        )

        model_config = getattr(
            underlying,
            "config",
            None,
        )

        model_max_length = getattr(
            model_config,
            "max_position_embeddings",
            None,
        )

        if (
            model_max_length
            is not None
            and model_max_length > 0
            and (
                prompt_length
                + response_length
                > model_max_length
            )
        ):

            available_prompt_tokens = (
                model_max_length
                - response_length
            )

            if (
                available_prompt_tokens
                <= 0
            ):

                raise ValueError(
                    "Response is too long for "
                    "the model context window."
                )

            prompt_ids = (
                prompt_ids[
                    -available_prompt_tokens:
                ]
            )

            prompt_length = int(
                prompt_ids.numel()
            )

        input_ids = (
            torch.cat(
                [
                    prompt_ids,
                    response_ids,
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        attention_mask = (
            torch.ones_like(
                input_ids
            )
        )

        outputs = (
            self.model(
                input_ids=(
                    input_ids
                ),
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
        )

        logits = (
            outputs.logits
        )

        # --------------------------------------------------------------
        # Causal alignment:
        #
        # logits[t] predicts token[t + 1]
        #
        # Therefore the first response token is predicted by the final
        # prompt position.
        # --------------------------------------------------------------

        response_logits = (
            logits[
                0,
                (
                    prompt_length
                    - 1
                ):
                (
                    prompt_length
                    - 1
                    + response_length
                ),
                :,
            ]
        )

        if (
            response_logits.shape[0]
            != response_length
        ):

            raise RuntimeError(
                "Unexpected causal-logit alignment. "
                f"Expected {response_length} response logits, "
                f"received {response_logits.shape[0]}."
            )

        log_probs = (
            torch.log_softmax(
                response_logits,
                dim=-1,
            )
        )

        response_ids = (
            response_ids.to(
                self.device
            )
        )

        token_log_probs = (
            log_probs
            .gather(
                dim=-1,
                index=(
                    response_ids
                    .unsqueeze(-1)
                ),
            )
            .squeeze(-1)
        )

        return token_log_probs

    # ========================================================================
    # CONVENIENCE GENERATION
    # ========================================================================

    def generate_one(
        self,
        prompt: str,
        temperature: float = 0.8,
        generation_index: int = 0,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        turn: int = 0,
        generation_type: str = "",
    ) -> GenerationResult:

        results = self.generate(
            prompts=[
                prompt
            ],
            temperature=(
                temperature
            ),
            generation_indices=[
                generation_index
            ],
            metadata=metadata,
            turn=turn,
            generation_type=(
                generation_type
            ),
        )

        if not results:

            raise RuntimeError(
                "generate_one() produced no result."
            )

        return results[0]

    def generate_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.8,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        turn: int = 0,
        generation_type: str = "",
    ) -> List[GenerationResult]:

        if n <= 0:

            raise ValueError(
                "n must be greater than 0."
            )

        return self.generate(
            prompts=[
                prompt
            ] * n,
            temperature=(
                temperature
            ),
            generation_indices=list(
                range(n)
            ),
            metadata=metadata,
            turn=turn,
            generation_type=(
                generation_type
            ),
        )

    # ========================================================================
    # MODEL ACCESS
    # ========================================================================

    def get_model(
        self,
    ):

        return self.model

    def get_underlying_model(
        self,
    ):

        return (
            self._underlying_model()
        )

    def get_tokenizer(
        self,
    ):

        return self.tokenizer

    def get_device(
        self,
    ) -> torch.device:

        return self.device

    def get_role(
        self,
    ) -> str:

        return self.role

    def get_adapter_name(
        self,
    ) -> str:

        return self.adapter_name

    def get_generation_config(
        self,
    ) -> Dict[str, Any]:

        return {
            "role": (
                self.role
            ),
            "adapter_name": (
                self.adapter_name
            ),
            "model_name": (
                self.model_name
            ),
            "max_input_tokens": (
                self.max_input_tokens
            ),
            "max_new_tokens": (
                self.max_new_tokens
            ),
            "top_p": (
                self.top_p
            ),
            "do_sample": (
                self.do_sample
            ),
            "generation_batch_size": (
                self.generation_batch_size
            ),
            "device": str(
                self.device
            ),
            "shared_backbone": bool(
                getattr(
                    config,
                    "USE_SHARED_BACKBONE",
                    True,
                )
            ),
        }

    # ========================================================================
    # TRAINABLE-PARAMETER DIAGNOSTICS
    # ========================================================================

    def get_trainable_parameter_count(
        self,
    ) -> Tuple[int, int]:
        """
        Return:

            trainable_parameter_count,
            total_parameter_count

        Parameter sharing means this is primarily diagnostic.
        """

        self._activate_adapter()

        trainable = 0
        total = 0

        for parameter in (
            self.model.parameters()
        ):

            count = int(
                parameter.numel()
            )

            total += count

            if parameter.requires_grad:

                trainable += count

        return (
            trainable,
            total,
        )

    # ========================================================================
    # TRAINING / EVALUATION STATE
    # ========================================================================

    def train(
        self,
    ) -> None:

        self._activate_adapter()

        self.model.train()

    def eval(
        self,
    ) -> None:

        self._activate_adapter()

        self.model.eval()

    def is_training(
        self,
    ) -> bool:

        return (
            self._model_training_state()
        )