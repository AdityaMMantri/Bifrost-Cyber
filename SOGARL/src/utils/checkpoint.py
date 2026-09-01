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

    def __init__(
        self,
        checkpoint_root: str | Path,
        logger=None,
    ) -> None:

        self.checkpoint_root = Path(
            checkpoint_root
        )

        self.logger = logger

        FileUtils.ensure_directory(
            self.checkpoint_root
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(
        self,
        checkpoint_name: str,
        red_model=None,
        blue_model=None,
        red_optimizer=None,
        blue_optimizer=None,
        red_reference_model=None,
        blue_reference_model=None,
        red_trainer=None,
        blue_trainer=None,
        epoch: int = 0,
        episode: int = 0,
        global_step: int = 0,
        best_metric: Optional[float] = None,
        trainer_state: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Path:

        checkpoint_dir = (
            self.checkpoint_root
            / checkpoint_name
        )

        if checkpoint_dir.exists():

            raise FileExistsError(
                f"Checkpoint already exists: "
                f"{checkpoint_dir}"
            )

        FileUtils.ensure_directory(
            checkpoint_dir
        )

        # --------------------------------------------------------------
        # Model adapters
        # --------------------------------------------------------------

        if red_model is not None:

            self._save_model(
                red_model,
                checkpoint_dir
                / "red_adapter",
            )

        if blue_model is not None:

            self._save_model(
                blue_model,
                checkpoint_dir
                / "blue_adapter",
            )

        # --------------------------------------------------------------
        # Frozen reference policies
        # --------------------------------------------------------------

        if red_reference_model is not None:

            self._save_reference_model(
                red_reference_model,
                checkpoint_dir
                / "reference_red.pt",
            )

        if blue_reference_model is not None:

            self._save_reference_model(
                blue_reference_model,
                checkpoint_dir
                / "reference_blue.pt",
            )

        # --------------------------------------------------------------
        # Trainer state
        # --------------------------------------------------------------

        red_trainer_state = (
            self._get_trainer_state(
                red_trainer
            )
        )

        blue_trainer_state = (
            self._get_trainer_state(
                blue_trainer
            )
        )

        # --------------------------------------------------------------
        # Backward-compatible optimizer arguments
        # --------------------------------------------------------------

        if (
            red_trainer_state is None
            and red_optimizer is not None
        ):

            red_trainer_state = {
                "optimizer": (
                    red_optimizer.state_dict()
                )
            }

        if (
            blue_trainer_state is None
            and blue_optimizer is not None
        ):

            blue_trainer_state = {
                "optimizer": (
                    blue_optimizer.state_dict()
                )
            }

        training_state = {
            "epoch": int(epoch),
            "episode": int(episode),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "red_trainer": (
                red_trainer_state
            ),
            "blue_trainer": (
                blue_trainer_state
            ),
        }

        if trainer_state is not None:

            training_state[
                "trainer_state"
            ] = trainer_state

        torch.save(
            training_state,
            checkpoint_dir
            / "trainer_state.pt",
        )

        # --------------------------------------------------------------
        # RNG
        # --------------------------------------------------------------

        self._save_rng_state(
            checkpoint_dir
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        info = {
            "checkpoint_name": (
                checkpoint_name
            ),
            "epoch": int(epoch),
            "episode": int(episode),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "has_red_adapter": (
                red_model is not None
            ),
            "has_blue_adapter": (
                blue_model is not None
            ),
            "has_red_reference": (
                red_reference_model is not None
            ),
            "has_blue_reference": (
                blue_reference_model is not None
            ),
            "has_red_trainer": (
                red_trainer_state is not None
            ),
            "has_blue_trainer": (
                blue_trainer_state is not None
            ),
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

        FileUtils.write_json(
            checkpoint_dir
            / "checkpoint_info.json",
            info,
        )

        if self.logger:

            self.logger.checkpoint(
                f"Saved checkpoint: "
                f"{checkpoint_dir}"
            )

        return checkpoint_dir

    # ==================================================================
    # LOAD
    # ==================================================================

    def load(
        self,
        checkpoint_path: str | Path,
        red_model=None,
        blue_model=None,
        red_optimizer=None,
        blue_optimizer=None,
        red_reference_model=None,
        blue_reference_model=None,
        red_trainer=None,
        blue_trainer=None,
        restore_rng: bool = True,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        require_training_state: bool = False,
    ) -> Dict[str, Any]:

        checkpoint_dir = Path(
            checkpoint_path
        )

        if not checkpoint_dir.is_dir():

            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint_dir}"
            )

        state_path = (
            checkpoint_dir
            / "trainer_state.pt"
        )

        if not state_path.exists():

            raise FileNotFoundError(
                f"Trainer state missing: "
                f"{state_path}"
            )

        state = torch.load(
            state_path,
            map_location=map_location,
            weights_only=False,
        )

        # --------------------------------------------------------------
        # Model adapters
        # --------------------------------------------------------------

        if red_model is not None:

            self._load_model(
                red_model,
                checkpoint_dir
                / "red_adapter",
                strict=strict,
            )

        if blue_model is not None:

            self._load_model(
                blue_model,
                checkpoint_dir
                / "blue_adapter",
                strict=strict,
            )

        # --------------------------------------------------------------
        # Frozen reference policies
        # --------------------------------------------------------------

        self._load_reference_model(
            model=red_reference_model,
            path=(
                checkpoint_dir
                / "reference_red.pt"
            ),
            map_location=map_location,
            strict=strict,
            required=(
                require_training_state
            ),
            role="Red",
        )

        self._load_reference_model(
            model=blue_reference_model,
            path=(
                checkpoint_dir
                / "reference_blue.pt"
            ),
            map_location=map_location,
            strict=strict,
            required=(
                require_training_state
            ),
            role="Blue",
        )

        # --------------------------------------------------------------
        # Trainer states
        # --------------------------------------------------------------

        red_state = state.get(
            "red_trainer"
        )

        blue_state = state.get(
            "blue_trainer"
        )

        if red_trainer is not None:

            if red_state is None:

                if require_training_state:

                    raise RuntimeError(
                        "Red trainer state is missing "
                        "from the checkpoint."
                    )

            else:

                red_trainer.load_state_dict(
                    red_state
                )

        elif (
            red_optimizer is not None
            and red_state is not None
            and "optimizer" in red_state
        ):

            red_optimizer.load_state_dict(
                red_state[
                    "optimizer"
                ]
            )

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

                blue_trainer.load_state_dict(
                    blue_state
                )

        elif (
            blue_optimizer is not None
            and blue_state is not None
            and "optimizer" in blue_state
        ):

            blue_optimizer.load_state_dict(
                blue_state[
                    "optimizer"
                ]
            )

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

            self._load_rng_state(
                checkpoint_dir
            )

        if self.logger:

            self.logger.checkpoint(
                f"Loaded checkpoint: "
                f"{checkpoint_dir}"
            )

        return {
            "trainer_state": state,
            "red_trainer_state": red_state,
            "blue_trainer_state": blue_state,
            "red_adapter_path": (
                checkpoint_dir
                / "red_adapter"
            ),
            "blue_adapter_path": (
                checkpoint_dir
                / "blue_adapter"
            ),
            "red_reference_path": (
                checkpoint_dir
                / "reference_red.pt"
            ),
            "blue_reference_path": (
                checkpoint_dir
                / "reference_blue.pt"
            ),
            "checkpoint_info": (
                self._safe_get_info(
                    checkpoint_dir
                )
            ),
        }

    # ==================================================================
    # TRAINER STATE
    # ==================================================================

    @staticmethod
    def _get_trainer_state(
        trainer,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if trainer is None:

            return None

        if not hasattr(
            trainer,
            "state_dict",
        ):

            raise TypeError(
                "Trainer must expose state_dict()."
            )

        return trainer.state_dict()

    # ==================================================================
    # REFERENCE MODEL
    # ==================================================================

    @staticmethod
    def _save_reference_model(
        model: torch.nn.Module,
        path: Path,
    ) -> None:

        state = {
            key: value.detach()
            .cpu()
            .clone()
            for key, value
            in model.state_dict().items()
        }

        torch.save(
            state,
            path,
        )

    @staticmethod
    def _load_reference_model(
        model: Optional[
            torch.nn.Module
        ],
        path: Path,
        map_location: str | torch.device,
        strict: bool,
        required: bool,
        role: str,
    ) -> None:

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

        model.load_state_dict(
            state,
            strict=strict,
        )

        model.eval()

        for parameter in (
            model.parameters()
        ):

            parameter.requires_grad = False

    # ==================================================================
    # MODEL
    # ==================================================================

    @staticmethod
    def _save_model(
        model: torch.nn.Module,
        output_dir: Path,
    ) -> None:

        FileUtils.ensure_directory(
            output_dir
        )

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

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model checkpoint missing: "
                f"{model_path}"
            )

        # --------------------------------------------------------------
        # If the model is already a PEFT model, load the saved adapter
        # weights into its existing adapter instead of creating a
        # second adapter.
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

            try:

                from safetensors.torch import (
                    load_file,
                )

                if state_path.exists():

                    state = load_file(
                        str(state_path)
                    )

                else:

                    state = torch.load(
                        bin_state_path,
                        map_location="cpu",
                        weights_only=False,
                    )

                model.load_state_dict(
                    state,
                    strict=False,
                )

                return

            except ImportError:

                if bin_state_path.exists():

                    state = torch.load(
                        bin_state_path,
                        map_location="cpu",
                        weights_only=False,
                    )

                    model.load_state_dict(
                        state,
                        strict=False,
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
    def _save_rng_state(
        checkpoint_dir: Path,
    ) -> None:

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():

            state["cuda"] = (
                torch.cuda.get_rng_state_all()
            )

        torch.save(
            state,
            checkpoint_dir
            / "rng_state.pt",
        )

    @staticmethod
    def _load_rng_state(
        checkpoint_dir: Path,
    ) -> None:

        rng_path = (
            checkpoint_dir
            / "rng_state.pt"
        )

        if not rng_path.exists():

            return

        state = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )

        if "python" in state:

            random.setstate(
                state["python"]
            )

        if "numpy" in state:

            np.random.set_state(
                state["numpy"]
            )

        if "torch" in state:

            torch.set_rng_state(
                state["torch"]
            )

        if (
            "cuda" in state
            and torch.cuda.is_available()
        ):

            torch.cuda.set_rng_state_all(
                state["cuda"]
            )

    # ==================================================================
    # LATEST
    # ==================================================================

    def find_latest(
        self,
    ) -> Optional[Path]:

        if not self.checkpoint_root.exists():

            return None

        checkpoints = [
            path
            for path
            in self.checkpoint_root.iterdir()
            if (
                path.is_dir()
                and (
                    path
                    / "checkpoint_info.json"
                ).exists()
            )
        ]

        if not checkpoints:

            return None

        return max(
            checkpoints,
            key=lambda path: (
                self._checkpoint_step(
                    path
                ),
                path.stat().st_mtime,
            ),
        )

    # ==================================================================
    # EXISTS
    # ==================================================================

    def exists(
        self,
        checkpoint_name: str,
    ) -> bool:

        return (
            self.checkpoint_root
            / checkpoint_name
        ).is_dir()

    # ==================================================================
    # INFO
    # ==================================================================

    def get_info(
        self,
        checkpoint_path: str | Path,
    ) -> Dict[str, Any]:

        checkpoint_dir = Path(
            checkpoint_path
        )

        info_path = (
            checkpoint_dir
            / "checkpoint_info.json"
        )

        if not info_path.exists():

            raise FileNotFoundError(
                f"Checkpoint metadata missing: "
                f"{info_path}"
            )

        return FileUtils.read_json(
            info_path
        )

    @staticmethod
    def _safe_get_info(
        checkpoint_dir: Path,
    ) -> Optional[
        Dict[str, Any]
    ]:

        info_path = (
            checkpoint_dir
            / "checkpoint_info.json"
        )

        if not info_path.exists():

            return None

        return FileUtils.read_json(
            info_path
        )

    # ==================================================================
    # DELETE
    # ==================================================================

    def delete(
        self,
        checkpoint_name: str,
    ) -> bool:

        checkpoint_dir = (
            self.checkpoint_root
            / checkpoint_name
        )

        if not checkpoint_dir.exists():

            return False

        shutil.rmtree(
            checkpoint_dir
        )

        if self.logger:

            self.logger.checkpoint(
                f"Deleted checkpoint: "
                f"{checkpoint_dir}"
            )

        return True

    # ==================================================================
    # INTERNAL
    # ==================================================================

    @staticmethod
    def _checkpoint_step(
        checkpoint_path: Path,
    ) -> int:

        info_path = (
            checkpoint_path
            / "checkpoint_info.json"
        )

        try:

            info = FileUtils.read_json(
                info_path
            )

            return int(
                info.get(
                    "global_step",
                    0,
                )
            )

        except Exception:

            return 0