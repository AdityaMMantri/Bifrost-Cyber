"""
checkpoint.py

Checkpoint management for SOGARL.

Stores:
    - Red adapter
    - Blue adapter
    - Red optimizer/trainer state
    - Blue optimizer/trainer state
    - Frozen Red reference policy
    - Frozen Blue reference policy
    - RNG state
    - Training metadata
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from src.utils.file_utils import FileUtils


class CheckpointManager:

    def __init__(self,checkpoint_root: str | Path,logger=None) -> None:
        self.checkpoint_root = Path(checkpoint_root)
        self.logger = logger
        FileUtils.ensure_directory(self.checkpoint_root)

    # SAVE
    def save(self,checkpoint_name: str,red_model=None,blue_model=None,red_optimizer=None,blue_optimizer=None,
             red_reference_model=None,blue_reference_model=None,red_trainer=None,blue_trainer=None,epoch: int = 0,episode: int = 0,
             global_step: int = 0,best_metric: Optional[float] = None,trainer_state: Optional[Dict[str, Any]] = None) -> Path:
        
        checkpoint_dir = (self.checkpoint_root/checkpoint_name)
        if checkpoint_dir.exists():
            raise FileExistsError(f"Checkpoint already exists: "f"{checkpoint_dir}")

        FileUtils.ensure_directory(checkpoint_dir)
        # Model adapters
        if red_model is not None:
            self._save_model(red_model,checkpoint_dir/"red_adapter")
        if blue_model is not None:
            self._save_model(blue_model,checkpoint_dir/"blue_adapter")
        # Frozen reference policies
        if red_reference_model is not None:
            self._save_reference_model(red_reference_model,checkpoint_dir/"reference_red.pt")
        if blue_reference_model is not None:
            self._save_reference_model(blue_reference_model,checkpoint_dir/"reference_blue.pt")
        # Trainer state
        red_trainer_state = (self._get_trainer_state(red_trainer))
        blue_trainer_state = (self._get_trainer_state(blue_trainer))
        # Backward-compatible optimizer arguments
        if (red_trainer_state is None and red_optimizer is not None):
            red_trainer_state = {"optimizer": (red_optimizer.state_dict())}
        if (blue_trainer_state is None and blue_optimizer is not None):
            blue_trainer_state = {"optimizer": (blue_optimizer.state_dict())}

        training_state = {
            "epoch": int(epoch),
            "episode": int(episode),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "red_trainer": (red_trainer_state),
            "blue_trainer": (blue_trainer_state)}

        if trainer_state is not None:
            training_state["trainer_state"] = trainer_state

        torch.save(training_state,checkpoint_dir/"trainer_state.pt")
        # RNG
        self._save_rng_state(checkpoint_dir)
        # Metadata
        info = {
            "checkpoint_name": (checkpoint_name),
            "epoch": int(epoch),
            "episode": int(episode),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "has_red_adapter": (red_model is not None),
            "has_blue_adapter": (blue_model is not None),
            "has_red_reference": (red_reference_model is not None),
            "has_blue_reference": (blue_reference_model is not None),
            "has_red_trainer": (red_trainer_state is not None),
            "has_blue_trainer": (blue_trainer_state is not None),
            "has_red_optimizer": (
                red_optimizer is not None
                or red_trainer_state is not None
            ),
            "has_blue_optimizer": (
                blue_optimizer is not None
                or blue_trainer_state is not None
            ),
            "format_version": 2,
        }

        FileUtils.write_json(checkpoint_dir/"checkpoint_info.json",info)
        if self.logger:
            self.logger.checkpoint(
                f"Saved checkpoint: "
                f"{checkpoint_dir}"
            )
        return checkpoint_dir

    # ==================================================================
    # LOAD
    # ==================================================================

    def load(self,checkpoint_path: str | Path,red_model=None,blue_model=None,red_optimizer=None,blue_optimizer=None,
             red_reference_model=None,blue_reference_model=None,red_trainer=None,blue_trainer=None,restore_rng: bool = True,
             map_location: str | torch.device = "cpu",strict: bool = True,require_training_state: bool = False) -> Dict[str, Any]:

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint_dir}"
            )
        state_path = (checkpoint_dir/"trainer_state.pt")
        if not state_path.exists():
            raise FileNotFoundError(
                f"Trainer state missing: "
                f"{state_path}")
        state = torch.load(
            state_path,
            map_location=map_location,
            weights_only=False,
        )

        # Model adapters
        if red_model is not None:
            self._load_model(red_model,checkpoint_dir/"red_adapter",strict=strict)
        if blue_model is not None:
            self._load_model(blue_model,checkpoint_dir/"blue_adapter",strict=strict)
        # Frozen reference policies
        self._load_reference_model(
            model=red_reference_model,
            path=(checkpoint_dir/"reference_red.pt"),
            map_location=map_location,
            strict=strict,
            required=(require_training_state),
            role="Red"
        )

        self._load_reference_model(
            model=blue_reference_model,
            path=(checkpoint_dir/"reference_blue.pt"),
            map_location=map_location,
            strict=strict,
            required=(require_training_state),
            role="Blue"
        )
        # Trainer states
        red_state = state.get("red_trainer")
        blue_state = state.get("blue_trainer")
        if red_trainer is not None:
            if red_state is None:
                if require_training_state:
                    raise RuntimeError(
                        "Red trainer state is missing "
                        "from the checkpoint."
                    )
            else:
                red_trainer.load_state_dict(red_state)
        elif (
            red_optimizer is not None
            and red_state is not None
            and "optimizer" in red_state
        ):
            red_optimizer.load_state_dict(red_state["optimizer"])

        elif (
            require_training_state
            and red_optimizer is not None
        ):

            raise RuntimeError(
                "Red optimizer state is missing "
                "from the checkpoint."
            )

        if blue_trainer is not None:
            if blue_state is None:
                if require_training_state:
                    raise RuntimeError(
                        "Blue trainer state is missing "
                        "from the checkpoint."
                    )
            else:
                blue_trainer.load_state_dict(blue_state)

        elif (
            blue_optimizer is not None
            and blue_state is not None
            and "optimizer" in blue_state
        ):

            blue_optimizer.load_state_dict(blue_state["optimizer"])

        elif (
            require_training_state
            and blue_optimizer is not None
        ):

            raise RuntimeError(
                "Blue optimizer state is missing "
                "from the checkpoint."
            )

        # --------------------------------------------------------------
        # RNG
        # --------------------------------------------------------------

        if restore_rng:
            self._load_rng_state(checkpoint_dir)
        if self.logger:
            self.logger.checkpoint(
                f"Loaded checkpoint: "
                f"{checkpoint_dir}"
            )
        return {
            "trainer_state": state,
            "red_trainer_state": red_state,
            "blue_trainer_state": blue_state,
            "red_adapter_path": (checkpoint_dir/"red_adapter"),
            "blue_adapter_path": (checkpoint_dir/"blue_adapter"),
            "red_reference_path": (checkpoint_dir/"reference_red.pt"),
            "blue_reference_path": (checkpoint_dir/"reference_blue.pt"),
            "checkpoint_info": (self._safe_get_info(checkpoint_dir))
        }

    # ==================================================================
    # TRAINER STATE
    # ==================================================================

    @staticmethod
    def _get_trainer_state(trainer) -> Optional[Dict[str, Any]]:
        if trainer is None:
            return None
        if not hasattr(trainer,"state_dict"):
            raise TypeError("Trainer must expose state_dict().")
        return trainer.state_dict()

    # ==================================================================
    # REFERENCE MODEL
    # ==================================================================

    @staticmethod
    def _save_reference_model(model: torch.nn.Module,path: Path) -> None:
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(state,path)

    @staticmethod
    def _load_reference_model(model: Optional[torch.nn.Module],path: Path,map_location: str | torch.device,\
                              strict: bool,required: bool,role: str) -> None:
        """
        Restore a shared-base frozen reference adapter.

        _SharedBaseReferenceModel stores only the reference LoRA adapter
        state.  Its state_dict keys belong to the underlying PEFT model,
        while the wrapper itself is not the object that should receive
        those keys directly.

        The reference adapter name can also differ after reconstruction
        (for example when another reference adapter already exists), so
        checkpoint keys are remapped to the currently instantiated
        reference adapter before loading.
        """

        if model is None:
            return

        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"{role} reference policy is "
                    f"missing from checkpoint: "
                    f"{path}"
                )
            return

        state = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(state, dict):
            raise TypeError(
                f"{role} reference checkpoint must contain "
                f"a state-dict, got {type(state).__name__}."
            )

        # --------------------------------------------------------------
        # _SharedBaseReferenceModel exposes the actual shared PEFT model
        # through _model and the currently allocated reference adapter
        # through _reference_adapter.
        # --------------------------------------------------------------

        shared_model = getattr(
            model,
            "_model",
            None,
        )

        reference_adapter = getattr(
            model,
            "_reference_adapter",
            None,
        )

        if (
            shared_model is not None
            and reference_adapter is not None
            and hasattr(
                shared_model,
                "state_dict",
            )
        ):
            current_state = shared_model.state_dict()

            loaded_state = {}

            # Checkpoint keys were saved from the reference wrapper and
            # therefore contain the old reference adapter name.
            old_reference_names = set()

            for key in state:
                marker = "."
                if marker in key:
                    parts = key.split(".")
                    for index, part in enumerate(parts):
                        if part.startswith(
                            "__sogarl_reference__"
                        ):
                            old_reference_names.add(part)

            # Usually there is exactly one such name.  Use the marker
            # found in the checkpoint; otherwise the current name is
            # already the correct name.
            old_reference_name = (
                next(iter(old_reference_names))
                if old_reference_names
                else reference_adapter
            )

            for key, value in state.items():
                target_key = key.replace(
                    f".{old_reference_name}.",
                    f".{reference_adapter}.",
                    1,
                )

                if target_key not in current_state:
                    if strict:
                        raise RuntimeError(
                            f"{role} reference checkpoint key "
                            f"does not exist in the current model: "
                            f"{target_key}"
                        )
                    continue

                loaded_state[target_key] = value

            if not loaded_state:
                raise RuntimeError(
                    f"No {role} reference adapter parameters "
                    "could be matched to the current shared PEFT model."
                )

            missing, unexpected = shared_model.load_state_dict(
                loaded_state,
                strict=False,
            )

            # Only the reference-adapter keys matter here.  PEFT/base
            # parameters are intentionally absent from this checkpoint.
            reference_marker = (
                f".{reference_adapter}."
            )

            missing_reference = [
                key
                for key in missing
                if reference_marker in key
            ]

            unexpected_reference = [
                key
                for key in unexpected
                if reference_marker in key
            ]

            if strict and (
                missing_reference
                or unexpected_reference
            ):
                raise RuntimeError(
                    f"{role} reference adapter restore failed. "
                    f"Missing: {missing_reference}; "
                    f"Unexpected: {unexpected_reference}"
                )

            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad = False

            return

        # --------------------------------------------------------------
        # Generic reference-model fallback.
        #
        # This keeps compatibility with explicitly supplied standalone
        # reference models that are not _SharedBaseReferenceModel.
        # --------------------------------------------------------------

        incompatible = model.load_state_dict(
            state,
            strict=strict,
        )

        if strict and (
            getattr(
                incompatible,
                "missing_keys",
                [],
            )
            or getattr(
                incompatible,
                "unexpected_keys",
                [],
            )
        ):
            raise RuntimeError(
                f"{role} reference policy restore reported "
                f"missing/unexpected keys: "
                f"{incompatible}"
            )

        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

    # ==================================================================
    # MODEL
    # ==================================================================

    @staticmethod
    def _unwrap_shared_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Return the underlying PEFT model when model is a SOGARL role view.

        _RoleModelView keeps the shared PEFT model in _shared_model.
        Standalone PEFT/generic models are returned unchanged.
        """
        shared_model = getattr(model,"_shared_model",None)

        if (shared_model is not None and isinstance(shared_model,torch.nn.Module)):
            return shared_model
        return model

    @staticmethod
    def _save_model(model: torch.nn.Module,output_dir: Path) -> None:
        """
        Save ONLY the adapter belonging to this role.

        Red and Blue share one PEFT backbone.  Calling
        save_pretrained() on the shared PEFT model without selecting the
        adapter can save whichever adapter happens to be active at the
        time of checkpointing.

        We therefore explicitly select the role adapter when PEFT is
        available.  The base model is never copied into the checkpoint.
        """
        FileUtils.ensure_directory(output_dir)
        shared_model = (CheckpointManager._unwrap_shared_model(model))
        adapter_name = getattr(model,"adapter_name",None)
        if (adapter_name is None and hasattr(shared_model,"peft_config")):

            active = getattr(shared_model,"active_adapter",None)
            if isinstance(active,(list, tuple)):
                adapter_name = (active[0] if active else None)
            else:
                adapter_name = active

        if (hasattr(shared_model,"save_pretrained")
            and hasattr(shared_model,"peft_config")):
            if (adapter_name is None or adapter_name not in shared_model.peft_config):
                raise RuntimeError(
                    "Cannot save role adapter: "
                    "the requested adapter is not present "
                    "in the shared PEFT model."
                )

            # PEFT supports selected_adapters so the checkpoint contains
            # exactly this role's adapter, regardless of the currently
            # active adapter.
            try:
                shared_model.save_pretrained(
                    output_dir,
                    selected_adapters=[
                        adapter_name
                    ],
                )
            except TypeError:
                # Compatibility with older PEFT versions that do not
                # expose selected_adapters.
                previous = getattr(
                    shared_model,
                    "active_adapter",
                    None,
                )

                try:
                    shared_model.set_adapter(adapter_name)
                    shared_model.save_pretrained(output_dir)
                finally:
                    if previous is not None:
                        shared_model.set_adapter(previous)

            return

        if hasattr(
            model,
            "save_pretrained",
        ):
            model.save_pretrained(
                output_dir
            )
            return

        torch.save(
            model.state_dict(),
            output_dir
            / "model.pt",
        )

    @staticmethod
    def _load_model(
        model: torch.nn.Module,
        model_path: Path,
        strict: bool = True,
    ) -> None:
        """
        Restore a role's LoRA adapter into an already-created shared PEFT
        model.

        The base model is intentionally NOT loaded from the checkpoint.

        PEFT's adapter-aware state loader is used instead of calling
        load_state_dict() on _RoleModelView, because the latter does not
        understand PEFT adapter checkpoint key semantics.
        """

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint missing: "
                f"{model_path}"
            )

        shared_model = (
            CheckpointManager._unwrap_shared_model(
                model
            )
        )

        adapter_name = getattr(
            model,
            "adapter_name",
            None,
        )

        # --------------------------------------------------------------
        # PEFT adapter checkpoint
        # --------------------------------------------------------------

        state_path = (
            model_path
            / "adapter_model.safetensors"
        )

        bin_state_path = (
            model_path
            / "adapter_model.bin"
        )

        if (
            state_path.exists()
            or bin_state_path.exists()
        ):
            if not hasattr(
                shared_model,
                "peft_config",
            ):
                raise RuntimeError(
                    "Checkpoint contains a PEFT adapter, but the "
                    "current model is not a PEFT model."
                )

            if (
                adapter_name is None
                or adapter_name
                not in shared_model.peft_config
            ):
                raise RuntimeError(
                    "Checkpoint adapter cannot be restored because "
                    "the target role adapter is not present in the "
                    "current shared PEFT model."
                )

            if state_path.exists():
                try:
                    from safetensors.torch import (
                        load_file,
                    )

                    state = load_file(
                        str(state_path),
                    )

                except ImportError as exc:
                    if not bin_state_path.exists():
                        raise RuntimeError(
                            "safetensors is required to load "
                            f"{state_path}."
                        ) from exc

                    state = torch.load(
                        bin_state_path,
                        map_location="cpu",
                        weights_only=False,
                    )

            else:
                state = torch.load(
                    bin_state_path,
                    map_location="cpu",
                    weights_only=False,
                )

            # ----------------------------------------------------------
            # Preferred PEFT-aware loader.
            # ----------------------------------------------------------

            try:
                from peft.utils.save_and_load import (
                    set_peft_model_state_dict,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "PEFT is required to restore a PEFT adapter "
                    "checkpoint."
                ) from exc

            incompatible = (
                set_peft_model_state_dict(
                    shared_model,
                    state,
                    adapter_name=adapter_name,
                )
            )

            if strict and incompatible is not None:
                missing = getattr(
                    incompatible,
                    "missing_keys",
                    [],
                )

                unexpected = getattr(
                    incompatible,
                    "unexpected_keys",
                    [],
                )

                if missing or unexpected:
                    raise RuntimeError(
                        "PEFT adapter restore failed for "
                        f"'{adapter_name}'. "
                        f"Missing: {missing}; "
                        f"Unexpected: {unexpected}"
                    )

            # Restore the role as active only after loading.  The next
            # Generator operation will also explicitly activate it.
            if hasattr(
                shared_model,
                "set_adapter",
            ):
                shared_model.set_adapter(
                    adapter_name
                )

            return

        # --------------------------------------------------------------
        # Generic PyTorch fallback
        # --------------------------------------------------------------

        state_path = (
            model_path
            / "model.pt"
        )

        if not state_path.exists():
            raise FileNotFoundError(
                f"No model state found in: "
                f"{model_path}"
            )

        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            state,
            strict=strict,
        )

    # ==================================================================
    # RNG
    # ==================================================================

    @staticmethod
    def _save_rng_state(checkpoint_dir: Path) -> None:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = (torch.cuda.get_rng_state_all())
        torch.save(state,checkpoint_dir/"rng_state.pt")

    @staticmethod
    def _load_rng_state(checkpoint_dir: Path) -> None:
        rng_path = (checkpoint_dir/"rng_state.pt")
        if not rng_path.exists():
            return
        state = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )

        if "python" in state:
            random.setstate(state["python"])

        if "numpy" in state:
            np.random.set_state(state["numpy"])

        if "torch" in state:
            torch.set_rng_state(state["torch"])

        if ("cuda" in state and torch.cuda.is_available()):
            torch.cuda.set_rng_state_all(state["cuda"])

    # ==================================================================
    # LATEST
    # ==================================================================

    def find_latest(self) -> Optional[Path]:
        if not self.checkpoint_root.exists():
            return None
        checkpoints = [path for path in self.checkpoint_root.iterdir()
                       if (path.is_dir() and (path/"checkpoint_info.json").exists())]
        if not checkpoints:
            return None

        return max(checkpoints,key=lambda path: (self._checkpoint_step(path),path.stat().st_mtime))

    # ==================================================================
    # EXISTS
    # ==================================================================

    def exists(self,checkpoint_name: str) -> bool:
        return (self.checkpoint_root/checkpoint_name).is_dir()

    # ==================================================================
    # INFO
    # ==================================================================

    def get_info(self,checkpoint_path: str | Path) -> Dict[str, Any]:
        checkpoint_dir = Path(checkpoint_path)
        info_path = (checkpoint_dir/"checkpoint_info.json")
        if not info_path.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata missing: "
                f"{info_path}")
        return FileUtils.read_json(info_path)

    @staticmethod
    def _safe_get_info(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        info_path = (checkpoint_dir/"checkpoint_info.json")
        if not info_path.exists():
            return None
        return FileUtils.read_json(info_path)

    # ==================================================================
    # DELETE
    # ==================================================================

    def delete(self,checkpoint_name: str) -> bool:
        checkpoint_dir = (self.checkpoint_root/checkpoint_name)
        if not checkpoint_dir.exists():
            return False
        shutil.rmtree(checkpoint_dir)

        if self.logger:
            self.logger.checkpoint(
                f"Deleted checkpoint: "
                f"{checkpoint_dir}")
        return True

    # ==================================================================
    # INTERNAL
    # ==================================================================

    @staticmethod
    def _checkpoint_step(checkpoint_path: Path) -> int:
        info_path = (checkpoint_path/"checkpoint_info.json")
        try:
            info = FileUtils.read_json(info_path)
            return int(info.get("global_step",0))
        except Exception:
            return 0