"""
grpo.py

Core GRPO optimization utilities for SOGARL.

Responsibilities:
    - advantage normalization
    - policy probability ratios
    - clipped policy objective
    - frozen reference-policy KL penalty
    - GRPO loss
    - optimizer updates
    - gradient accumulation
    - trainer state management

Does NOT:
    - generate responses
    - build prompts
    - evaluate vulnerabilities
    - calculate rewards
    - select scenarios
    - manage episodes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from configs import config


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GRPOConfig:

    clip_epsilon: float = (
        config.GRPO_CLIP_EPSILON
    )

    kl_coefficient: float = (
        config.KL_COEFFICIENT
    )

    advantage_epsilon: float = (
        config.ADVANTAGE_EPSILON
    )

    max_grad_norm: float = (
        config.MAX_GRAD_NORM
    )

    gradient_accumulation_steps: int = (
        config.GRADIENT_ACCUMULATION_STEPS
    )

    def __post_init__(self) -> None:

        if self.clip_epsilon <= 0:
            raise ValueError(
                "clip_epsilon must be greater than zero."
            )

        if self.kl_coefficient < 0:
            raise ValueError(
                "kl_coefficient cannot be negative."
            )

        if self.advantage_epsilon <= 0:
            raise ValueError(
                "advantage_epsilon must be greater than zero."
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be greater than zero."
            )

        if (
            self.gradient_accumulation_steps
            <= 0
        ):
            raise ValueError(
                "gradient_accumulation_steps must be "
                "greater than zero."
            )


# ============================================================================
# LOSS RESULT
# ============================================================================

@dataclass
class GRPOLoss:

    total_loss: torch.Tensor

    policy_loss: torch.Tensor

    kl_loss: torch.Tensor

    mean_advantage: torch.Tensor

    mean_ratio: torch.Tensor

    clipped_fraction: torch.Tensor

    valid_token_fraction: torch.Tensor

    def to_dict(
        self,
    ) -> Dict[str, float]:

        return {
            "total_loss": float(
                self.total_loss.detach().cpu()
            ),
            "policy_loss": float(
                self.policy_loss.detach().cpu()
            ),
            "kl_loss": float(
                self.kl_loss.detach().cpu()
            ),
            "mean_advantage": float(
                self.mean_advantage.detach().cpu()
            ),
            "mean_ratio": float(
                self.mean_ratio.detach().cpu()
            ),
            "clipped_fraction": float(
                self.clipped_fraction.detach().cpu()
            ),
            "valid_token_fraction": float(
                self.valid_token_fraction.detach().cpu()
            ),
        }


# ============================================================================
# GRPO OBJECTIVE
# ============================================================================

class GRPOOptimizer:

    def __init__(
        self,
        grpo_config: Optional[
            GRPOConfig
        ] = None,
        logger=None,
    ) -> None:

        self.config = (
            grpo_config
            if grpo_config is not None
            else GRPOConfig()
        )

        self.logger = logger

    # ==================================================================
    # ADVANTAGES
    # ==================================================================

    def normalize_advantages(
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:

        if advantages.numel() == 0:
            raise ValueError(
                "Advantages cannot be empty."
            )

        advantages = advantages.float()

        mean = advantages.mean()

        std = advantages.std(
            unbiased=False
        )

        return (
            advantages - mean
        ) / (
            std
            + self.config.advantage_epsilon
        )

    # ==================================================================
    # TENSOR PREPARATION
    # ==================================================================

    @staticmethod
    def _as_tensor(
        values: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        device: Optional[
            torch.device
        ] = None,
    ) -> torch.Tensor:

        if isinstance(
            values,
            torch.Tensor,
        ):

            tensor = values

            if tensor.ndim == 1:

                tensor = tensor.unsqueeze(
                    0
                )

            if tensor.ndim != 2:

                raise ValueError(
                    "Log-probability tensors must "
                    "have shape [batch, tokens]."
                )

            if device is not None:

                tensor = tensor.to(
                    device
                )

            return tensor

        values = list(values)

        if not values:

            raise ValueError(
                "Log-probability sequence "
                "cannot be empty."
            )

        tensors = []

        for value in values:

            if not isinstance(
                value,
                torch.Tensor,
            ):

                value = torch.as_tensor(
                    value,
                    dtype=torch.float32,
                )

            value = value.flatten()

            if device is not None:

                value = value.to(
                    device
                )

            tensors.append(
                value
            )

        max_length = max(
            tensor.numel()
            for tensor in tensors
        )

        if max_length <= 0:

            raise ValueError(
                "All log-probability tensors "
                "are empty."
            )

        result = torch.zeros(
            (
                len(tensors),
                max_length,
            ),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

        for index, tensor in enumerate(
            tensors
        ):

            result[
                index,
                :tensor.numel(),
            ] = tensor

        return result

    @staticmethod
    def _create_mask(
        values: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:

        if isinstance(
            values,
            torch.Tensor,
        ):

            if values.ndim == 1:

                values = values.unsqueeze(
                    0
                )

            return torch.ones(
                shape,
                dtype=torch.bool,
                device=device,
            )

        values = list(values)

        mask = torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        )

        for row, tensor in enumerate(
            values
        ):

            if isinstance(
                tensor,
                torch.Tensor,
            ):

                length = tensor.numel()

            else:

                length = len(tensor)

            mask[
                row,
                :min(
                    length,
                    shape[1],
                ),
            ] = True

        return mask

    @staticmethod
    def _prepare_advantages(
        advantages: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:

        advantages = advantages.to(
            device=device,
            dtype=torch.float32,
        )

        if advantages.ndim == 0:

            advantages = advantages.reshape(
                1
            )

        if advantages.ndim == 2:

            if advantages.shape[1] != 1:

                raise ValueError(
                    "Advantages must have shape [batch]."
                )

            advantages = advantages.squeeze(
                1
            )

        if advantages.ndim != 1:

            raise ValueError(
                "Advantages must have shape [batch]."
            )

        if (
            advantages.numel()
            != batch_size
        ):

            raise ValueError(
                "Number of advantages must match "
                "number of candidates."
            )

        return advantages

    # ==================================================================
    # POLICY RATIO
    # ==================================================================

    def policy_ratio(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:

        if (
            current_log_probs.shape
            != old_log_probs.shape
        ):

            raise ValueError(
                "Current and old log-probabilities "
                "must have identical shapes."
            )

        return torch.exp(
            current_log_probs
            - old_log_probs.detach()
        )

    # ==================================================================
    # CLIPPED POLICY OBJECTIVE
    # ==================================================================

    def clipped_policy_loss(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        if (
            current_log_probs.shape
            != old_log_probs.shape
        ):

            raise ValueError(
                "Current and old log-probabilities "
                "must have identical shapes."
            )

        batch_size, token_count = (
            current_log_probs.shape
        )

        advantages = (
            self._prepare_advantages(
                advantages,
                batch_size,
                current_log_probs.device,
            )
        )

        if response_mask is None:

            response_mask = torch.ones(
                (
                    batch_size,
                    token_count,
                ),
                dtype=torch.bool,
                device=current_log_probs.device,
            )

        response_mask = response_mask.to(
            device=current_log_probs.device,
            dtype=torch.bool,
        )

        if (
            response_mask.shape
            != current_log_probs.shape
        ):

            raise ValueError(
                "response_mask must have the same "
                "shape as log probabilities."
            )

        ratio = self.policy_ratio(
            current_log_probs,
            old_log_probs,
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0
            - self.config.clip_epsilon,
            1.0
            + self.config.clip_epsilon,
        )

        token_advantages = (
            advantages.unsqueeze(1)
            .expand_as(ratio)
        )

        unclipped = (
            ratio
            * token_advantages
        )

        clipped = (
            clipped_ratio
            * token_advantages
        )

        surrogate = torch.minimum(
            unclipped,
            clipped,
        )

        mask = response_mask.float()

        denominator = (
            mask.sum()
            .clamp_min(1.0)
        )

        policy_loss = -(
            surrogate * mask
        ).sum() / denominator

        clipped_fraction = (
            (
                (ratio != clipped_ratio)
                & response_mask
            )
            .float()
            .sum()
            / denominator
        )

        return (
            policy_loss,
            ratio,
            clipped_fraction,
        )

    # ==================================================================
    # REFERENCE KL
    # ==================================================================

    def kl_divergence(
        self,
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        response_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        if (
            current_log_probs.shape
            != reference_log_probs.shape
        ):

            raise ValueError(
                "Current and reference "
                "log-probabilities must have "
                "identical shapes."
            )

        if response_mask is None:

            response_mask = torch.ones(
                current_log_probs.shape,
                dtype=torch.bool,
                device=current_log_probs.device,
            )

        response_mask = response_mask.to(
            device=current_log_probs.device,
            dtype=torch.bool,
        )

        if (
            response_mask.shape
            != current_log_probs.shape
        ):

            raise ValueError(
                "response_mask must have the same "
                "shape as log probabilities."
            )

        reference = (
            reference_log_probs.detach()
        )

        log_ratio = (
            current_log_probs
            - reference
        )

        # --------------------------------------------------------------
        # Sampled KL estimator:
        #
        # KL ≈ exp(log π_current - log π_ref)
        #      - (log π_current - log π_ref)
        #      - 1
        #
        # This is non-negative and has its minimum at zero.
        # --------------------------------------------------------------

        kl = (
            torch.exp(log_ratio)
            - log_ratio
            - 1.0
        )

        mask = response_mask.float()

        denominator = (
            mask.sum()
            .clamp_min(1.0)
        )

        return (
            kl * mask
        ).sum() / denominator

    # ==================================================================
    # COMPLETE LOSS
    # ==================================================================

    def compute_loss(
        self,
        current_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        old_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        reference_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        advantages: torch.Tensor,
        normalize_advantages: bool = True,
        response_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> GRPOLoss:

        current = self._as_tensor(
            current_log_probs
        )

        old = self._as_tensor(
            old_log_probs,
            device=current.device,
        )

        reference = self._as_tensor(
            reference_log_probs,
            device=current.device,
        )

        if (
            current.shape
            != old.shape
            or current.shape
            != reference.shape
        ):

            raise ValueError(
                "Current, old, and reference "
                "log-probabilities must have "
                "identical shapes."
            )

        if response_mask is None:

            response_mask = (
                self._create_mask(
                    current_log_probs,
                    current.shape,
                    current.device,
                )
            )

        if normalize_advantages:

            advantages = (
                self.normalize_advantages(
                    advantages
                )
            )

        (
            policy_loss,
            ratio,
            clipped_fraction,
        ) = self.clipped_policy_loss(
            current_log_probs=current,
            old_log_probs=old,
            advantages=advantages,
            response_mask=response_mask,
        )

        kl_loss = self.kl_divergence(
            current_log_probs=current,
            reference_log_probs=reference,
            response_mask=response_mask,
        )

        total_loss = (
            policy_loss
            + (
                self.config.kl_coefficient
                * kl_loss
            )
        )

        mask = response_mask.float()

        valid_tokens = (
            mask.sum()
        )

        valid_token_fraction = (
            valid_tokens
            / response_mask.numel()
        )

        mean_ratio = (
            (
                ratio * mask
            ).sum()
            / valid_tokens.clamp_min(1.0)
        )

        return GRPOLoss(
            total_loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            mean_advantage=advantages.mean(),
            mean_ratio=mean_ratio,
            clipped_fraction=clipped_fraction,
            valid_token_fraction=(
                valid_token_fraction
            ),
        )


# ============================================================================
# SHARED-BASE REFERENCE POLICY
# ============================================================================

class _SharedBaseReferenceModel(torch.nn.Module):
    """
    Frozen reference policy that shares the quantized base model with the
    trainable PEFT model.

    Only the LoRA adapter weights are duplicated. The large base model is
    NOT deep-copied. This is important for QLoRA/4-bit training on GPUs with
    limited VRAM.

    The wrapper keeps the public model call interface intact: callers can
    continue to use reference_model(...) exactly as before.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:

        super().__init__()

        self._model = model
        self._reference_adapter = None
        self._original_adapter = None

        # --------------------------------------------------------------
        # PEFT models can host multiple adapters on the same frozen base.
        # Create a second adapter containing the initial LoRA weights.
        # --------------------------------------------------------------

        if not hasattr(model, "peft_config"):

            raise ValueError(
                "GRPO reference sharing requires a PEFT/LoRA model. "
                "The supplied model does not expose peft_config."
            )

        peft_config = model.peft_config

        if not peft_config:

            raise ValueError(
                "GRPO reference sharing requires at least one PEFT adapter."
            )

        active = getattr(
            model,
            "adapter_name",
            None,
        )

        if active is None:

            active = getattr(
                model,
                "active_adapter",
                None,
            )

        if isinstance(active, (list, tuple)):

            if not active:

                raise ValueError(
                    "The PEFT model has no active adapter."
                )

            active = active[0]

        if active is None:

            active = next(
                iter(peft_config.keys())
            )

        self._original_adapter = active

        reference_name = "__sogarl_reference__"

        # Avoid a collision if this trainer/model is reused.
        suffix = 0
        candidate_name = reference_name

        while candidate_name in peft_config:

            suffix += 1
            candidate_name = (
                f"{reference_name}_{suffix}"
            )

        reference_name = candidate_name
        self._reference_adapter = reference_name

        # --------------------------------------------------------------
        # Make sure the correct role's adapter is the one active on the
        # shared backbone before adding a new adapter. PEFT re-marks
        # which adapters are trainable based on whichever adapter is
        # currently active when add_adapter() runs, so if the wrong
        # adapter is active here, the real role adapter's requires_grad
        # gets silently cleared.
        # --------------------------------------------------------------

        if hasattr(model, "set_adapter"):

            model.set_adapter(
                active
            )

        # --------------------------------------------------------------
        # Add a LoRA adapter sharing the same base model.
        # --------------------------------------------------------------

        model.add_adapter(
            reference_name,
            peft_config[active],
        )

        # --------------------------------------------------------------
        # Copy ONLY adapter parameters.
        #
        # The base model remains shared.
        # --------------------------------------------------------------

        state = model.state_dict()

        copied = 0

        for key, value in state.items():

            source_token = (
                f".{active}."
            )

            target_token = (
                f".{reference_name}."
            )

            if source_token not in key:
                continue

            target_key = key.replace(
                source_token,
                target_token,
                1,
            )

            if target_key not in state:
                continue

            target = state[target_key]

            with torch.no_grad():

                target.copy_(
                    value.detach()
                )

            copied += 1

        if copied == 0:

            raise RuntimeError(
                "Could not copy the initial LoRA adapter weights "
                "into the GRPO reference adapter."
            )

        # --------------------------------------------------------------
        # Reference adapter is frozen.
        # --------------------------------------------------------------

        for name, parameter in model.named_parameters():

            if (
                f".{reference_name}."
                in name
            ):

                parameter.requires_grad = False

    def forward(
        self,
        *args,
        **kwargs,
    ):
        """
        Execute the frozen reference adapter while restoring the trainable
        adapter immediately afterward.
        """

        previous = getattr(
            self._model,
            "active_adapter",
            self._original_adapter,
        )

        try:

            self._model.set_adapter(
                self._reference_adapter
            )

            with torch.no_grad():

                return self._model(
                    *args,
                    **kwargs,
                )

        finally:

            self._model.set_adapter(
                previous
            )

    def train(
        self,
        mode: bool = True,
    ):
        """
        Preserve nn.Module train/eval semantics without changing the wrapped
        model's active policy.
        """

        self.training = mode

        return self

    def eval(
        self,
    ):
        self.training = False

        return self

    def state_dict(
        self,
        *args,
        **kwargs,
    ):
        """
        Return only the reference adapter state rather than duplicating the
        complete base-model state.
        """

        state = self._model.state_dict(
            *args,
            **kwargs,
        )

        result = {}

        token = (
            f".{self._reference_adapter}."
        )

        for key, value in state.items():

            if token in key:

                result[key] = (
                    value.detach()
                    .cpu()
                    .clone()
                )

        return result

    def parameters(
        self,
        recurse: bool = True,
    ):
        """
        Expose only the frozen reference adapter's own parameters.

        NOTE: this must NOT delegate to self._model.parameters(), since
        self._model is the role-specific view (e.g. Red's) and its
        parameters() returns that role's live, trainable adapter weights.
        Iterating those here would let callers (e.g. freezing the
        reference model) accidentally set requires_grad=False on the
        real trainable adapter instead of the reference clone.
        """

        token = (
            f".{self._reference_adapter}."
        )

        for (
            name,
            parameter,
        ) in self._model.named_parameters(
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
        token = (
            f".{self._reference_adapter}."
        )

        for (
            name,
            parameter,
        ) in self._model.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):

            if token in name:

                yield (
                    name,
                    parameter,
                )

    def __getattr__(
        self,
        name,
    ):

        if name in {
            "_model",
            "_reference_adapter",
            "_original_adapter",
        }:

            return super().__getattr__(
                name
            )

        return getattr(
            self._model,
            name,
        )


# ============================================================================
# GRPO TRAINER
# ============================================================================

class GRPOTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        grpo_config: Optional[
            GRPOConfig
        ] = None,
        weight_decay: float = (
            config.WEIGHT_DECAY
        ),
        reference_model: Optional[
            torch.nn.Module
        ] = None,
        logger=None,
    ) -> None:

        if learning_rate <= 0:

            raise ValueError(
                "learning_rate must be greater than zero."
            )

        self.model = model

        self.logger = logger

        self.grpo = GRPOOptimizer(
            grpo_config=grpo_config,
            logger=logger,
        )

        # --------------------------------------------------------------
        # Frozen reference policy
        #
        # IMPORTANT:
        # Do NOT deepcopy an 8B model here. With Red and Blue QLoRA models,
        # a full deepcopy creates another complete base-model copy and can
        # exhaust a 14-16 GB GPU.
        #
        # When the supplied policy is a PEFT/LoRA model, the reference
        # policy shares the same quantized base and duplicates only the
        # small LoRA adapter weights.
        #
        # An explicitly supplied reference_model is still respected.
        # --------------------------------------------------------------

        if reference_model is None:

            self.reference_model = (
                _SharedBaseReferenceModel(
                    model
                )
            )

        else:

            self.reference_model = (
                reference_model
            )

        self._freeze_reference_model()

        # --------------------------------------------------------------
        # Only trainable parameters belong in the optimizer.
        # This is important for LoRA/PEFT models.
        # --------------------------------------------------------------

        trainable_parameters = [
            parameter
            for parameter
            in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:

            raise ValueError(
                "No trainable parameters found "
                "for GRPO."
            )

        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.update_count = 0

        self.accumulation_count = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

    # ==================================================================
    # REFERENCE MODEL
    # ==================================================================

    def _freeze_reference_model(
        self,
    ) -> None:

        self.reference_model.eval()

        for parameter in (
            self.reference_model.parameters()
        ):

            parameter.requires_grad = False

    def get_reference_model(
        self,
    ) -> torch.nn.Module:

        return self.reference_model

    def reference_state_dict(
        self,
    ) -> Dict[str, torch.Tensor]:

        return {
            key: value.detach()
            .cpu()
            .clone()
            for key, value
            in self.reference_model.state_dict().items()
        }

    # ==================================================================
    # UPDATE
    # ==================================================================

    def update(
        self,
        current_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        old_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        reference_log_probs: (
            torch.Tensor
            | Sequence[torch.Tensor]
        ),
        advantages: torch.Tensor,
        normalize_advantages: bool = False,
        response_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Dict[str, float]:

        self.model.train()

        loss_result = self.grpo.compute_loss(
            current_log_probs=(
                current_log_probs
            ),
            old_log_probs=(
                old_log_probs
            ),
            reference_log_probs=(
                reference_log_probs
            ),
            advantages=advantages,
            normalize_advantages=(
                normalize_advantages
            ),
            response_mask=response_mask,
        )

        accumulation_steps = (
            self.grpo
            .config
            .gradient_accumulation_steps
        )

        scaled_loss = (
            loss_result.total_loss
            / accumulation_steps
        )

        scaled_loss.backward()

        self.accumulation_count += 1

        should_step = (
            self.accumulation_count
            >= accumulation_steps
        )

        gradient_norm = 0.0

        if should_step:

            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grpo
                    .config
                    .max_grad_norm,
                )
            )

            self.optimizer.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.accumulation_count = 0

            self.update_count += 1

        metrics = {
            "loss": float(
                loss_result.total_loss
                .detach()
                .cpu()
            ),
            "policy_loss": float(
                loss_result.policy_loss
                .detach()
                .cpu()
            ),
            "kl_loss": float(
                loss_result.kl_loss
                .detach()
                .cpu()
            ),
            "mean_advantage": float(
                loss_result
                .mean_advantage
                .detach()
                .cpu()
            ),
            "mean_ratio": float(
                loss_result
                .mean_ratio
                .detach()
                .cpu()
            ),
            "clipped_fraction": float(
                loss_result
                .clipped_fraction
                .detach()
                .cpu()
            ),
            "valid_token_fraction": float(
                loss_result
                .valid_token_fraction
                .detach()
                .cpu()
            ),
            "gradient_norm": gradient_norm,
            "optimizer_step": float(
                should_step
            ),
            "update_count": float(
                self.update_count
            ),
            "accumulation_count": float(
                self.accumulation_count
            ),
        }

        if self.logger:

            self.logger.info(
                "GRPO | "
                f"loss={metrics['loss']:.6f} | "
                f"policy={metrics['policy_loss']:.6f} | "
                f"KL={metrics['kl_loss']:.6f} | "
                f"adv={metrics['mean_advantage']:.6f} | "
                f"ratio={metrics['mean_ratio']:.6f}"
            )

        return metrics

    # ==================================================================
    # FLUSH
    # ==================================================================

    def flush(
        self,
    ) -> Optional[float]:

        if self.accumulation_count == 0:

            return None

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grpo.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.accumulation_count = 0

        self.update_count += 1

        return gradient_norm

    # ==================================================================
    # MODEL MODES
    # ==================================================================

    def train(
        self,
    ) -> None:

        self.model.train()

        self.reference_model.eval()

    def eval(
        self,
    ) -> None:

        self.model.eval()

        self.reference_model.eval()

    # ==================================================================
    # STATE
    # ==================================================================

    def state_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "optimizer": (
                self.optimizer.state_dict()
            ),
            "update_count": (
                self.update_count
            ),
            "accumulation_count": (
                self.accumulation_count
            ),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
    ) -> None:

        if "optimizer" in state:

            self.optimizer.load_state_dict(
                state["optimizer"]
            )

        self.update_count = int(
            state.get(
                "update_count",
                0,
            )
        )

        self.accumulation_count = int(
            state.get(
                "accumulation_count",
                0,
            )
        )

        self._freeze_reference_model()

    # ==================================================================
    # LEARNING RATE
    # ==================================================================

    def get_learning_rate(
        self,
    ) -> float:

        if not self.optimizer.param_groups:

            return 0.0

        return float(
            self.optimizer
            .param_groups[0]["lr"]
        )

    def set_learning_rate(
        self,
        learning_rate: float,
    ) -> None:

        if learning_rate <= 0:

            raise ValueError(
                "learning_rate must be greater than zero."
            )

        for group in (
            self.optimizer.param_groups
        ):

            group["lr"] = learning_rate

    # ==================================================================
    # SUMMARY
    # ==================================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        total_parameters = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
            if parameter.requires_grad
        )

        reference_parameters = sum(
            parameter.numel()
            for parameter
            in self.reference_model.parameters()
        )

        return {
            "updates": (
                self.update_count
            ),
            "pending_accumulation": (
                self.accumulation_count
            ),
            "learning_rate": (
                self.get_learning_rate()
            ),
            "total_parameters": (
                total_parameters
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
            "reference_parameters": (
                reference_parameters
            ),
            "reference_frozen": True,
        }


# ============================================================================
# TRAINER FACTORIES
# ============================================================================

def create_red_trainer(
    model: torch.nn.Module,
    grpo_config: Optional[
        GRPOConfig
    ] = None,
    reference_model: Optional[
        torch.nn.Module
    ] = None,
    logger=None,
) -> GRPOTrainer:

    return GRPOTrainer(
        model=model,
        learning_rate=(
            config.LEARNING_RATE_RED
        ),
        grpo_config=grpo_config,
        weight_decay=config.WEIGHT_DECAY,
        reference_model=reference_model,
        logger=logger,
    )


def create_blue_trainer(
    model: torch.nn.Module,
    grpo_config: Optional[
        GRPOConfig
    ] = None,
    reference_model: Optional[
        torch.nn.Module
    ] = None,
    logger=None,
) -> GRPOTrainer:

    return GRPOTrainer(
        model=model,
        learning_rate=(
            config.LEARNING_RATE_BLUE
        ),
        grpo_config=grpo_config,
        weight_decay=config.WEIGHT_DECAY,
        reference_model=reference_model,
        logger=logger,
    )