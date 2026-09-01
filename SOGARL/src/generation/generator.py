"""
Generation and policy-scoring interface for SOGARL.

Responsibilities:
    - Load Red/Blue SFT LoRA policies
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

MODEL LOADING:

    1. Local merged model, if available
    2. Local base model + LoRA adapter, if available
    3. Hugging Face base model + LoRA adapter, otherwise

IMPORTANT:

    Red and Blue use separate LoRA adapters but share ONE base model.

    First policy:
        base model -> Red LoRA

    Second policy:
        SAME base model -> Blue LoRA

    The base model is therefore NOT loaded twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from configs import config
from src.data.models import GenerationResult


class _RoleModelView(torch.nn.Module):
    """
    Role-specific view over one shared PEFT model.

    The underlying quantized backbone is shared between Red and Blue.

    Each Generator receives a view exposing only its own LoRA adapter
    as trainable parameters.

    This preserves the existing Generator.model interface while
    preventing Red and Blue from loading independent copies of the
    large base model.
    """

    def __init__(
        self,
        model,
        adapter_name: str,
    ) -> None:

        super().__init__()

        self._shared_model = model

        self.adapter_name = adapter_name

    def _activate(
        self,
    ) -> None:

        if hasattr(
            self._shared_model,
            "set_adapter",
        ):

            self._shared_model.set_adapter(
                self.adapter_name
            )

        current_token = (
            f".{self.adapter_name}."
        )

        for (
            name,
            parameter,
        ) in self._shared_model.named_parameters():

            if current_token not in name:

                parameter.requires_grad = False

        current_count = 0

        for (
            name,
            parameter,
        ) in self._shared_model.named_parameters():

            if current_token in name:

                parameter.requires_grad = True

                current_count += 1

        if current_count == 0:

            raise RuntimeError(
                f"No parameters found for active "
                f"LoRA adapter '{self.adapter_name}'."
            )

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
            isinstance(
                name,
                str,
            )
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

    def state_dict(
        self,
        *args,
        **kwargs,
    ):

        return self._shared_model.state_dict(
            *args,
            **kwargs,
        )

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


class Generator:

    # ==================================================================
    # SHARED BACKBONE STATE
    # ==================================================================

    _shared_model = None

    _shared_tokenizer = None

    _shared_model_source = None

    _shared_device = None

    _shared_adapter_names = set()

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        role: str,
        model_name: str = "unknown",
        max_input_tokens: int = 6148,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        do_sample: bool = True,
        generation_batch_size: int = 1,
        logger=None,
    ) -> None:

        self.model = model

        self.tokenizer = tokenizer

        self.device = device

        self.role = role.lower()

        self.model_name = model_name

        self.adapter_name = self.role

        self.max_input_tokens = (
            max_input_tokens
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        self.top_p = top_p

        self.do_sample = do_sample

        self.generation_batch_size = (
            generation_batch_size
        )

        self.logger = logger

        self._validate_config()

        self._prepare_tokenizer()

    # ==================================================================
    # MODEL LOADING
    # ==================================================================

    @classmethod
    def from_config(
        cls,
        role: str,
        logger=None,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        adapter_path = (
            config.RED_ADAPTER_PATH
            if role == "red"
            else config.BLUE_ADAPTER_PATH
        )

        merged_model_path = (
            cls._find_merged_model(
                role=role,
                adapter_path=adapter_path,
            )
        )

        if merged_model_path is not None:

            if logger:

                logger.info(
                    f"Loading {role.capitalize()} "
                    f"policy from LOCAL MERGED MODEL"
                )

                logger.info(
                    f"Merged model : "
                    f"{merged_model_path}"
                )

            return cls.from_merged_model(
                role=role,
                model_path=merged_model_path,
                logger=logger,
            )

        return cls.from_adapter(
            role=role,
            adapter_path=adapter_path,
            logger=logger,
            is_trainable=True,
        )

    # ==================================================================
    # MERGED MODEL LOADING
    # ==================================================================

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
                f"does not exist:\n{model_path}"
            )

        if not model_path.is_dir():

            raise NotADirectoryError(
                f"{role.capitalize()} merged model "
                f"path is not a directory:\n"
                f"{model_path}"
            )

        if not (
            model_path / "config.json"
        ).is_file():

            raise FileNotFoundError(
                f"config.json was not found in "
                f"merged model:\n{model_path}"
            )

        device = cls._resolve_device()

        dtype = cls._resolve_dtype(
            device
        )

        if logger:

            logger.info(
                f"Loading {role.capitalize()} "
                f"merged model locally"
            )

            logger.info(
                f"Model  : {model_path}"
            )

            logger.info(
                f"Device : {device}"
            )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                str(model_path),
                use_fast=True,
                local_files_only=True,
            )
        )

        model = (
            AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        )

        model.to(device)

        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            role=role,
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

    # ==================================================================
    # FIND MERGED MODEL
    # ==================================================================

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

                return configured_path

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

    # ==================================================================
    # ADAPTER LOADING
    # ==================================================================

    @classmethod
    def from_adapter(
        cls,
        role: str,
        adapter_path: str | Path,
        logger=None,
        is_trainable: bool = True,
    ) -> "Generator":

        role = cls._validate_role(
            role
        )

        base_model_name = (
            config.BASE_MODEL_NAME
        )

        adapter_path = Path(
            adapter_path
        ).expanduser().resolve()

        if not adapter_path.exists():

            raise FileNotFoundError(
                f"{role.capitalize()} LoRA adapter "
                f"does not exist:\n{adapter_path}"
            )

        device = cls._resolve_device()

        dtype = cls._resolve_dtype(
            device
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

        if logger:

            logger.info(
                f"Loading {role.capitalize()} policy"
            )

            logger.info(
                f"Base model : "
                f"{model_source}"
            )

            logger.info(
                f"Adapter    : "
                f"{adapter_path}"
            )

            logger.info(
                f"Device     : "
                f"{device}"
            )

            if local_files_only:

                logger.info(
                    "Base model source: LOCAL"
                )

            else:

                logger.info(
                    "Base model source: "
                    "HUGGING FACE FALLBACK"
                )

        if (
            cls._shared_tokenizer is None
            or cls._shared_model_source
            != model_source
        ):

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_source,
                    use_fast=True,
                    local_files_only=(
                        local_files_only
                    ),
                )
            )

            cls._shared_tokenizer = (
                tokenizer
            )

        else:

            tokenizer = (
                cls._shared_tokenizer
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

        # ==============================================================
        # FIRST POLICY
        # ==============================================================

        first_load = (
            cls._shared_model is None
            or cls._shared_model_source
            != model_source
            or cls._shared_device
            != device
        )

        if first_load:

            model_load_kwargs = {
                "local_files_only": (
                    local_files_only
                ),
            }

            if config.USE_4BIT_MODEL:

                if device.type != "cuda":

                    raise RuntimeError(
                        "USE_4BIT_MODEL=True requires CUDA "
                        "because bitsandbytes 4-bit loading "
                        "is configured for the Kaggle GPU path."
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
                        bnb_4bit_compute_dtype=dtype,
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

                if logger:

                    logger.info(
                        "================================================"
                    )

                    logger.info(
                        "LOADING ONE SHARED BASE MODEL"
                    )

                    logger.info(
                        "Quantization: 4-bit NF4"
                    )

                    logger.info(
                        "================================================"
                    )

            else:

                model_load_kwargs[
                    "torch_dtype"
                ] = dtype

                if logger:

                    logger.info(
                        "================================================"
                    )

                    logger.info(
                        "LOADING ONE SHARED BASE MODEL"
                    )

                    logger.info(
                        "Quantization: disabled"
                    )

                    logger.info(
                        "================================================"
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
                PeftModel.from_pretrained(
                    base_model,
                    str(adapter_path),
                    adapter_name=role,
                    is_trainable=is_trainable,
                )
            )

            cls._shared_model = model

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
                    f"Shared backbone loaded with "
                    f"{role.capitalize()} LoRA."
                )

                logger.info(
                    "No second base-model load will occur "
                    "for the other policy."
                )

        # ==============================================================
        # SUBSEQUENT POLICY
        # ==============================================================

        else:

            model = cls._shared_model

            if role not in (
                cls._shared_adapter_names
            ):

                if logger:

                    logger.info(
                        "================================================"
                    )

                    logger.info(
                        f"REUSING SHARED BASE MODEL FOR "
                        f"{role.upper()}"
                    )

                    logger.info(
                        "Loading LoRA adapter only."
                    )

                    logger.info(
                        "NO base-model from_pretrained() call."
                    )

                    logger.info(
                        "================================================"
                    )

                model.load_adapter(
                    str(adapter_path),
                    adapter_name=role,
                    is_trainable=is_trainable,
                )

                cls._shared_adapter_names.add(
                    role
                )

                if logger:

                    logger.info(
                        f"{role.capitalize()} LoRA adapter "
                        f"attached to shared backbone."
                    )

            else:

                if logger:

                    logger.info(
                        f"{role.capitalize()} adapter already "
                        f"exists on shared backbone."
                    )

        model.set_adapter(
            role
        )

        model_view = _RoleModelView(
            model=model,
            adapter_name=role,
        )

        if is_trainable:

            model_view._activate()

        else:

            for parameter in (
                model_view.parameters()
            ):

                parameter.requires_grad = False

            model_view.eval()

        return cls(
            model=model_view,
            tokenizer=tokenizer,
            device=device,
            role=role,
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
    # LOCAL BASE MODEL RESOLUTION
    # ==================================================================

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
            path / "config.json"
        ).is_file():

            return None

        return path

    # ==================================================================
    # CHECKPOINT LOADING
    # ==================================================================

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

        checkpoint_path = Path(
            checkpoint_path
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
                f"adapter does not exist:\n"
                f"{adapter_path}"
            )

        if logger:

            logger.info(
                f"Loading {role.capitalize()} "
                f"policy from checkpoint"
            )

            logger.info(
                f"Checkpoint : "
                f"{checkpoint_path}"
            )

        return cls.from_adapter(
            role=role,
            adapter_path=adapter_path,
            logger=logger,
            is_trainable=is_trainable,
        )

    # ==================================================================
    # DEVICE / DTYPE
    # ==================================================================

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

            if not torch.cuda.is_bf16_supported():

                raise RuntimeError(
                    "USE_BF16=True but this CUDA "
                    "device does not support bfloat16."
                )

            return torch.bfloat16

        if config.USE_FP16:

            return torch.float16

        return torch.float32

    # ==================================================================
    # VALIDATION
    # ==================================================================

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
                "role must be either 'red' or 'blue'."
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
                "role must be either 'red' or 'blue'."
            )

        if self.max_input_tokens <= 0:

            raise ValueError(
                "max_input_tokens must be "
                "greater than 0."
            )

        if self.max_new_tokens <= 0:

            raise ValueError(
                "max_new_tokens must be "
                "greater than 0."
            )

        if not 0.0 < self.top_p <= 1.0:

            raise ValueError(
                "top_p must be in the range (0, 1]."
            )

        if self.generation_batch_size <= 0:

            raise ValueError(
                "generation_batch_size must be "
                "greater than 0."
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
                    "pad_token_id nor eos_token_id."
                )

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.tokenizer.padding_side = "left"

    # ==================================================================
    # GENERATION
    # ==================================================================

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

        if temperature <= 0:

            raise ValueError(
                "temperature must be greater than 0."
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

        results = []

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

            results.extend(
                self._generate_batch(
                    prompts=prompts[
                        start:end
                    ],
                    temperature=temperature,
                    generation_indices=(
                        generation_indices[
                            start:end
                        ]
                    ),
                    metadata=metadata,
                    turn=turn,
                    generation_type=(
                        generation_type
                    ),
                )
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

        self._activate_adapter()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_attention_mask=True,
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            **self._generation_kwargs(
                temperature
            ),
        )

        input_width = (
            inputs["input_ids"]
            .shape[1]
        )

        results = []

        for index, output in enumerate(
            outputs
        ):

            generated_tokens = (
                output[
                    input_width:
                ]
            )

            response = (
                self.tokenizer.decode(
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
                    "role": self.role,
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
                    response=response,
                    prompt=prompts[index],
                    model_name=self.model_name,
                    temperature=temperature,
                    generation_index=(
                        generation_indices[
                            index
                        ]
                    ),
                    turn=turn,
                    generation_type=(
                        generation_type
                    ),
                    metadata=result_metadata,
                )
            )

        return results

    # ==================================================================
    # GENERATION PARAMETERS
    # ==================================================================

    def _generation_kwargs(
        self,
        temperature: float,
    ) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # MEMORY FIX:
        #
        # The Kaggle GPU has ~14.5 GB VRAM. With a 4-bit Llama-3.1-8B
        # backbone and a 6148-token input context, generating 2048
        # tokens can require several additional GB for the KV cache.
        #
        # Keep the configured max_new_tokens API unchanged, but cap the
        # actual generation length at 512 for this constrained GPU.
        #
        # Input context remains controlled by MAX_INPUT_TOKENS.
        # --------------------------------------------------------------

        safe_max_new_tokens = min(
            self.max_new_tokens,
            512,
        )

        kwargs = {
            "max_new_tokens": safe_max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "use_cache": True,
        }

        if (
            self.tokenizer.eos_token_id
            is not None
        ):

            kwargs[
                "eos_token_id"
            ] = (
                self.tokenizer.eos_token_id
            )

        if self.do_sample:

            kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": self.top_p,
                }
            )

        return kwargs

    # ==================================================================
    # RESPONSE LOG PROBABILITIES
    # ==================================================================

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
                "prompts and responses must have "
                "the same length."
            )

        if not prompts:

            return []

        if require_grad:

            return [
                self._compute_single_log_probs(
                    prompt=prompt,
                    response=response,
                )
                for prompt, response
                in zip(
                    prompts,
                    responses,
                )
            ]

        with torch.no_grad():

            return [
                self._compute_single_log_probs(
                    prompt=prompt,
                    response=response,
                )
                for prompt, response
                in zip(
                    prompts,
                    responses,
                )
            ]

    def _compute_single_log_probs(
        self,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """
        Return log P(response token | previous tokens).

        Only response-token log probabilities are returned.
        Prompt-token probabilities are excluded.
        """

        self._activate_adapter()

        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        # --------------------------------------------------------------
        # MEMORY FIX:
        #
        # Keep response scoring bounded by the same 512-token generation
        # limit used above.
        # --------------------------------------------------------------

        response_tokens = self.tokenizer(
            response,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=min(
                self.max_new_tokens,
                512,
            ),
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

        if response_ids.numel() == 0:

            return torch.empty(
                0,
                device=self.device,
                dtype=torch.float32,
            )

        # --------------------------------------------------------------
        # The total sequence must fit inside max_input_tokens.
        # Reserve enough space for the complete response.
        # --------------------------------------------------------------

        available_prompt_tokens = (
            self.max_input_tokens
            - response_ids.numel()
        )

        if (
            available_prompt_tokens
            <= 0
        ):

            raise ValueError(
                "Response is too long for the "
                "configured context window."
            )

        if (
            prompt_ids.numel()
            > available_prompt_tokens
        ):

            prompt_ids = prompt_ids[
                -available_prompt_tokens:
            ]

        prompt_length = (
            prompt_ids.numel()
        )

        response_length = (
            response_ids.numel()
        )

        input_ids = torch.cat(
            [
                prompt_ids,
                response_ids,
            ],
            dim=0,
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        attention_mask = torch.ones_like(
            input_ids
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        # --------------------------------------------------------------
        # Causal LM alignment:
        #
        # logits[t] predicts token[t + 1]
        #
        # Therefore the first response token is predicted by
        # the last prompt token.
        # --------------------------------------------------------------

        response_logits = logits[
            0,
            prompt_length - 1:
            prompt_length - 1
            + response_length,
            :,
        ]

        log_probs = torch.log_softmax(
            response_logits,
            dim=-1,
        )

        response_ids = response_ids.to(
            self.device
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

    # ==================================================================
    # CONVENIENCE GENERATION
    # ==================================================================

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
            prompts=[prompt],
            temperature=temperature,
            generation_indices=[
                generation_index
            ],
            metadata=metadata,
            turn=turn,
            generation_type=(
                generation_type
            ),
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
            prompts=[prompt] * n,
            temperature=temperature,
            generation_indices=list(
                range(n)
            ),
            metadata=metadata,
            turn=turn,
            generation_type=(
                generation_type
            ),
        )

    # ==================================================================
    # MODEL ACCESS
    # ==================================================================

    def _activate_adapter(
        self,
    ) -> None:

        model = self.model

        if hasattr(
            model,
            "_activate",
        ):

            model._activate()

            return

        if hasattr(
            model,
            "_shared_model",
        ):

            shared = (
                model._shared_model
            )

            if hasattr(
                shared,
                "set_adapter",
            ):

                shared.set_adapter(
                    self.adapter_name
                )

    def get_model(
        self,
    ):

        return self.model

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

    def get_generation_config(
        self,
    ) -> Dict[str, Any]:

        return {
            "role": self.role,
            "model_name": self.model_name,
            "max_input_tokens": (
                self.max_input_tokens
            ),
            "max_new_tokens": (
                self.max_new_tokens
            ),
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "generation_batch_size": (
                self.generation_batch_size
            ),
            "device": str(
                self.device
            ),
        }

    # ==================================================================
    # TRAINING / EVALUATION STATE
    # ==================================================================

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

        return self.model.training