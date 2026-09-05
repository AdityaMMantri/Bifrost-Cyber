"""
episode_manager.py

Orchestrates one complete SOGARL episode.

Flow:

    Scenario
       ↓
    Red Turn 1
       ↓
    Oracle
       ↓
    Attack_best
       ↓
    Confidence Gate
       ↓
    Blue Turn 2
       ↓
    Oracle
       ↓
    Top-K
       ↓
    Red Turn 3
       ↓
    Oracle interaction verification
       ↓
    RewardManager
       ↓
    Red GRPO + Blue GRPO
       ↓
    Replay + Metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from configs import config

from src.data.models import (
    Scenario,
    ScenarioContext,
    GenerationResult,
    ScoredCandidate,
)

from src.data.scenario_loader import ScenarioLoader
from src.data.metadata_loader import MetadataLoader
from src.data.path_resolver import PathResolver
from src.data.code_loader import CodeLoader

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.interaction_checker import InteractionChecker

from src.rl.reward_manager import RewardManager
from src.rl.grpo import GRPOTrainer

from src.logging.replay_buffer import (
    ReplayBuffer,
    ReplayRecord,
)

from src.logging.metrics_logger import MetricsLogger


# ============================================================================
# EPISODE RESULT
# ============================================================================

@dataclass
class EpisodeResult:

    scenario_id: str

    red_candidates: List[ScoredCandidate]

    blue_candidates: List[ScoredCandidate]

    top_k_blue_candidates: List[ScoredCandidate]

    challenges: List[GenerationResult]

    interaction_results: List[Dict[str, Any]]

    red_update: Optional[Dict[str, float]]

    blue_update: Optional[Dict[str, float]]

    episode_metrics: Dict[str, Any]

    episode_id: str = ""

    epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scenario_id": self.scenario_id,
            "episode_id": self.episode_id,
            "epoch": self.epoch,

            "red_candidates": [
                self._candidate_to_dict(
                    candidate
                )
                for candidate in self.red_candidates
            ],

            "blue_candidates": [
                self._candidate_to_dict(
                    candidate
                )
                for candidate in self.blue_candidates
            ],

            "top_k_blue_candidates": [
                self._candidate_to_dict(
                    candidate
                )
                for candidate in self.top_k_blue_candidates
            ],

            "challenges": [
                self._generation_to_dict(
                    generation
                )
                for generation in self.challenges
            ],

            "interaction_results": (
                self.interaction_results
            ),

            "red_update": self.red_update,

            "blue_update": self.blue_update,

            "episode_metrics": self.episode_metrics,
        }

    @staticmethod
    def _candidate_to_dict(
        candidate: ScoredCandidate,
    ) -> Dict[str, Any]:

        metadata = {}

        for key, value in candidate.metadata.items():

            if isinstance(
                value,
                torch.Tensor,
            ):
                continue

            metadata[key] = value

        return {
            "response": candidate.response,

            "oracle_reward": (
                candidate.oracle_reward
            ),

            "candidate_index": (
                candidate.candidate_index
            ),

            "prompt": getattr(
                candidate,
                "prompt",
                "",
            ),

            "generation_type": getattr(
                candidate,
                "generation_type",
                "",
            ),

            "interaction_reward": (
                candidate.interaction_reward
            ),

            "oracle_advantage": (
                candidate.oracle_advantage
            ),

            "interaction_advantage": (
                candidate.interaction_advantage
            ),

            "final_advantage": (
                candidate.final_advantage
            ),

            "metadata": metadata,
        }

    @staticmethod
    def _generation_to_dict(
        generation: GenerationResult,
    ) -> Dict[str, Any]:

        metadata = {}

        for key, value in generation.metadata.items():

            if isinstance(
                value,
                torch.Tensor,
            ):
                continue

            metadata[key] = value

        return {
            "response": generation.response,

            "prompt": generation.prompt,

            "model_name": generation.model_name,

            "temperature": generation.temperature,

            "generation_index": (
                generation.generation_index
            ),

            "turn": generation.turn,

            "generation_type": (
                generation.generation_type
            ),

            "metadata": metadata,
        }


# ============================================================================
# EPISODE MANAGER
# ============================================================================

class EpisodeManager:

    def __init__(
        self,
        scenario_loader: ScenarioLoader,
        metadata_loader: MetadataLoader,
        path_resolver: PathResolver,
        code_loader: CodeLoader,
        prompt_builder: PromptBuilder,
        red_generator: Generator,
        blue_generator: Generator,
        oracle: Oracle,
        interaction_checker: Optional[
            InteractionChecker
        ],
        reward_manager: RewardManager,
        red_trainer: Optional[GRPOTrainer],
        blue_trainer: Optional[GRPOTrainer],
        replay_buffer: Optional[ReplayBuffer] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        logger=None,
    ) -> None:

        self.scenario_loader = scenario_loader
        self.metadata_loader = metadata_loader
        self.path_resolver = path_resolver
        self.code_loader = code_loader

        self.prompt_builder = prompt_builder

        self.red_generator = red_generator
        self.blue_generator = blue_generator

        self.oracle = oracle

        self.interaction_checker = (
            interaction_checker
        )

        self.reward_manager = reward_manager

        self.red_trainer = red_trainer
        self.blue_trainer = blue_trainer

        self.replay_buffer = replay_buffer
        self.metrics_logger = metrics_logger

        self.logger = logger

        self._episode_counter = 0

        self._reference_generators: Dict[
            str,
            Generator,
        ] = {}

    # ==================================================================
    # COMPLETE EPISODE
    # ==================================================================

    def run_episode(
        self,
        scenario: Scenario,
        training: bool = True,
        episode_id: Optional[str] = None,
        epoch: int = 0,
    ) -> EpisodeResult:

        self._episode_counter += 1

        if episode_id is None:

            episode_id = (
                f"episode_"
                f"{self._episode_counter:06d}"
            )

        if self.logger:

            self.logger.info(
                f"Starting {episode_id} | "
                f"Scenario: {scenario.scenario_id}"
            )

        # --------------------------------------------------------------
        # PREPARE SAFE MODEL CONTEXT
        # --------------------------------------------------------------

        context = self._prepare_context(
            scenario
        )

        # ==============================================================
        # TURN 1 — RED ATTACK
        # ==============================================================

        red_generations = (
            self._run_red_attack_turn(
                context
            )
        )

        red_candidates = (
            self._score_red_candidates(
                scenario,
                red_generations,
            )
        )

        red_category = self._scenario_category(
            scenario
        )

        for candidate in red_candidates:
            candidate.metadata[
                "scenario_category"
            ] = red_category

        # --------------------------------------------------------------
        # Red Turn 1 is trained ONLY using Oracle advantages.
        #
        # IMPORTANT:
        # The Red optimizer step is deliberately delayed until AFTER
        # Turn 3. This keeps Turn 1 and Turn 3 inside one episode under
        # the same Red policy parameters.
        #
        # Turn-3 interaction rewards still do NOT create a second Red
        # update in V1.
        # --------------------------------------------------------------

        red_update = None

        # ==============================================================
        # SELECT ATTACK_BEST + CONFIDENCE GATE
        # ==============================================================

        red_attack = (
            self._select_red_context(
                red_candidates
            )
        )

        # ==============================================================
        # TURN 2 — BLUE DEFENSE
        # ==============================================================

        blue_generations = (
            self._run_blue_defense_turn(
                context,
                red_attack,
            )
        )

        blue_candidates = (
            self._score_blue_candidates(
                scenario,
                blue_generations,
                red_attack,
            )
        )

        # ==============================================================
        # TOP-K BLUE DEFENSES
        # ==============================================================

        top_k_blue = self._select_top_k(
            blue_candidates
        )

        # ==============================================================
        # TURN 3 — RED CHALLENGES
        #
        # Exactly the selected Top-K defenses are challenged.
        # No challenge is generated for unselected Blue candidates.
        # ==============================================================

        challenges: List[
            GenerationResult
        ] = []

        interaction_results: List[
            Dict[str, Any]
        ] = []

        for blue_candidate in top_k_blue:

            challenge = (
                self._run_red_challenge(
                    context,
                    blue_candidate,
                )
            )

            challenges.append(
                challenge
            )

            interaction = (
                self._evaluate_interaction(
                    scenario,
                    blue_candidate,
                    challenge,
                )
            )

            interaction_results.append(
                interaction
            )

        # ==============================================================
        # APPLY TURN-3 INTERACTION REWARDS
        # ==============================================================

        self._apply_interaction_rewards(
            all_blue_candidates=blue_candidates,
            top_k_candidates=top_k_blue,
            interaction_results=interaction_results,
        )

        # ==============================================================
        # BLUE FINAL ADVANTAGE
        #
        # RewardManager:
        #
        #     A_oracle
        #         ↓
        #     A_interaction
        #         ↓
        #     alpha*A_oracle + beta*A_interaction
        #
        # Interaction normalization is performed ONLY over tested
        # Top-K candidates.
        # ==============================================================

        category = self._scenario_category(
            scenario
        )

        # Training updates category history after computing the current
        # episode's beta. Evaluation must remain read-only: test episodes
        # must not modify the category history used by RewardManager.
        if training:
            self.reward_manager.process_blue_candidates(
                candidates=blue_candidates,
                category=category,
            )
        else:
            self.reward_manager.calculate_final_advantages(
                candidates=blue_candidates,
                category=category,
            )

        # ==============================================================
        # RED GRPO
        #
        # Exactly one Red GRPO update per episode.
        #
        # It is intentionally performed only after Turn 3 has finished,
        # so Red Turn 1 and Red Turn 3 are generated by the same policy
        # version. The update still uses Turn-1 Oracle advantages only.
        # ==============================================================

        if (
            training
            and config.TRAIN_UPDATES_ENABLED
        ):

            self._prepare_training_log_probs(
                generator=self.red_generator,
                candidates=red_candidates,
                trainer=self.red_trainer,
            )

            red_update = self._update_red(
                red_candidates
            )

        # ==============================================================
        # BLUE GRPO
        #
        # Exactly one Blue GRPO update per episode.
        # ==============================================================

        blue_update = None

        if (
            training
            and config.TRAIN_UPDATES_ENABLED
        ):

            self._prepare_training_log_probs(
                generator=self.blue_generator,
                candidates=blue_candidates,
                trainer=self.blue_trainer,
            )

            blue_update = self._update_blue(
                blue_candidates
            )

        # ==============================================================
        # METRICS
        # ==============================================================

        episode_metrics = (
            self._build_episode_metrics(
                red_candidates=red_candidates,
                blue_candidates=blue_candidates,
                top_k_blue=top_k_blue,
                interaction_results=(
                    interaction_results
                ),
                red_update=red_update,
                blue_update=blue_update,
                confidence_gate_passed=(
                    red_attack is not None
                ),
            )
        )

        result = EpisodeResult(
            scenario_id=scenario.scenario_id,

            red_candidates=red_candidates,

            blue_candidates=blue_candidates,

            top_k_blue_candidates=top_k_blue,

            challenges=challenges,

            interaction_results=(
                interaction_results
            ),

            red_update=red_update,

            blue_update=blue_update,

            episode_metrics=episode_metrics,

            episode_id=episode_id,

            epoch=epoch,
        )

        self._log_episode(
            result
        )

        return result

    # ==================================================================
    # CONTEXT
    # ==================================================================

    def _prepare_context(
        self,
        scenario: Scenario,
    ) -> ScenarioContext:

        resolved_files = (
            self.path_resolver.resolve(
                scenario_path=(
                    scenario.scenario_path
                ),
                relevant_files=(
                    scenario.relevant_files
                ),
                noise_files=(
                    scenario.noise_files
                ),
            )
        )

        code_files = (
            self.code_loader.load(
                resolved_files
            )
        )

        return ScenarioContext(
            scenario=scenario,
            code_files=code_files,
        )

    @staticmethod
    def _scenario_category(
        scenario: Scenario,
    ) -> str:

        category = getattr(
            scenario,
            "category",
            None,
        )

        if category:

            return str(
                category
            )

        category = getattr(
            scenario,
            "attack_category",
            None,
        )

        if category:

            return str(
                category
            )

        return "unknown"

    # ==================================================================
    # TURN 1 — RED
    # ==================================================================

    def _run_red_attack_turn(
        self,
        context: ScenarioContext,
    ) -> List[GenerationResult]:

        prompt = (
            self.prompt_builder
            .build_red_attack_prompt(
                context
            )
        )

        return (
            self.red_generator.generate_n(
                prompt=prompt,
                n=config.NUM_ROLLOUTS,
                temperature=(
                    config.RED_TEMPERATURE
                ),
                metadata={
                    "turn": 1,
                    "generation_type": (
                        "red_attack"
                    ),
                },
                turn=1,
                generation_type="red_attack",
            )
        )

    # ==================================================================
    # RED SCORING
    # ==================================================================

    def _score_red_candidates(
        self,
        scenario: Scenario,
        generations: List[GenerationResult],
    ) -> List[ScoredCandidate]:

        candidates = []

        for generation in generations:

            oracle_result = (
                self.oracle.score_attack(
                    response=generation.response,
                    scenario=scenario,
                )
            )

            metadata = dict(
                generation.metadata
            )

            metadata.update(
                oracle_result.metadata
            )

            candidates.append(
                ScoredCandidate(
                    response=generation.response,

                    oracle_reward=(
                        oracle_result.reward
                    ),

                    candidate_index=(
                        generation.generation_index
                    ),

                    prompt=generation.prompt,

                    generation_type=(
                        generation.generation_type
                    ),

                    metadata=metadata,
                )
            )

        return (
            self.reward_manager
            .calculate_red_advantages(
                candidates
            )
        )

    # ==================================================================
    # RED SELECTION
    # ==================================================================

    def _select_red_context(
        self,
        candidates: List[ScoredCandidate],
    ) -> Optional[ScoredCandidate]:

        if not candidates:

            return None

        best = max(
            candidates,
            key=lambda candidate:
                candidate.oracle_reward,
        )

        if not config.USE_CONFIDENCE_GATE:

            return best

        return self._apply_confidence_gate(
            candidates,
            best,
        )

    def _apply_confidence_gate(
        self,
        candidates: List[ScoredCandidate],
        best_candidate: ScoredCandidate,
    ) -> Optional[ScoredCandidate]:

        if len(candidates) < 2:

            if (
                config
                .BLUE_USE_INDEPENDENT_FALLBACK
            ):

                return None

            return best_candidate

        # --------------------------------------------------------------
        # Category-relative confidence gate.
        #
        # Use historical category statistics rather than the current
        # rollout group. This prevents the gate from being defined by
        # the same eight candidates it is supposed to assess.
        #
        # Until enough category history exists, the gate is not applied;
        # the best Red candidate is passed through normally. This avoids
        # making an early-training decision from an undefined baseline.
        # --------------------------------------------------------------

        category = getattr(
            best_candidate,
            "metadata",
            {},
        ).get(
            "scenario_category"
        )

        if not category:
            # The category is attached by _select_red_context's caller
            # through candidate metadata when available. Fall back to the
            # existing rollout-relative behavior only when no category
            # information is available.
            rewards = torch.tensor(
                [
                    float(
                        candidate.oracle_reward
                    )
                    for candidate in candidates
                ],
                dtype=torch.float32,
            )

            mean = rewards.mean()
            std = rewards.std(
                unbiased=False
            )

        else:
            statistics = (
                self.reward_manager
                .get_category_statistics(
                    str(category)
                )
            )

            minimum_episodes = int(
                getattr(
                    config,
                    "MIN_CATEGORY_EPISODES_FOR_GATE",
                    20,
                )
            )

            if (
                statistics["episode_count"]
                < minimum_episodes
            ):
                return best_candidate

            mean = torch.tensor(
                statistics["mean_reward"],
                dtype=torch.float32,
            )
            std = torch.tensor(
                statistics["std_reward"],
                dtype=torch.float32,
            )

        threshold = (
            mean
            + (
                config
                .CONFIDENCE_STD_MULTIPLIER
                * std
            )
        )

        if (
            best_candidate.oracle_reward
            >= threshold
        ):

            return best_candidate

        if (
            config
            .BLUE_USE_INDEPENDENT_FALLBACK
        ):

            return None

        return best_candidate

    # ==================================================================
    # TURN 2 — BLUE
    # ==================================================================

    def _run_blue_defense_turn(
        self,
        context: ScenarioContext,
        red_attack: Optional[ScoredCandidate],
    ) -> List[GenerationResult]:

        attack_best = (
            red_attack.response
            if red_attack is not None
            else None
        )

        confidence_passed = (
            red_attack is not None
        )

        prompt = (
            self.prompt_builder
            .build_blue_defense_prompt(
                context=context,
                attack_best=attack_best,
                confidence_passed=(
                    confidence_passed
                ),
            )
        )

        return (
            self.blue_generator.generate_n(
                prompt=prompt,
                n=config.NUM_ROLLOUTS,
                temperature=(
                    config.BLUE_TEMPERATURE
                ),
                metadata={
                    "turn": 2,
                    "generation_type": (
                        "blue_defense"
                    ),
                    "confidence_passed": (
                        confidence_passed
                    ),
                },
                turn=2,
                generation_type="blue_defense",
            )
        )

    # ==================================================================
    # BLUE SCORING
    # ==================================================================

    def _score_blue_candidates(
        self,
        scenario: Scenario,
        generations: List[GenerationResult],
        red_attack: Optional[ScoredCandidate],
    ) -> List[ScoredCandidate]:

        attack_response = (
            red_attack.response
            if red_attack is not None
            else None
        )

        candidates = []

        for generation in generations:

            oracle_result = (
                self.oracle.score_defense(
                    response=generation.response,
                    scenario=scenario,
                    attack_response=(
                        attack_response
                    ),
                )
            )

            metadata = dict(
                generation.metadata
            )

            metadata.update(
                oracle_result.metadata
            )

            candidates.append(
                ScoredCandidate(
                    response=generation.response,

                    oracle_reward=(
                        oracle_result.reward
                    ),

                    candidate_index=(
                        generation.generation_index
                    ),

                    prompt=generation.prompt,

                    generation_type=(
                        generation.generation_type
                    ),

                    metadata=metadata,
                )
            )

        return (
            self.reward_manager
            .calculate_oracle_advantages(
                candidates
            )
        )

    # ==================================================================
    # TOP-K
    # ==================================================================

    def _select_top_k(
        self,
        candidates: List[ScoredCandidate],
    ) -> List[ScoredCandidate]:

        if not candidates:

            return []

        ranked = sorted(
            candidates,
            key=lambda candidate:
                candidate.oracle_reward,
            reverse=True,
        )

        return ranked[
            :config.TOP_K
        ]

    # ==================================================================
    # TURN 3 — RED CHALLENGE
    # ==================================================================

    def _run_red_challenge(
        self,
        context: ScenarioContext,
        blue_candidate: ScoredCandidate,
    ) -> GenerationResult:

        prompt = (
            self.prompt_builder
            .build_red_challenge_prompt(
                context=context,
                blue_defense=(
                    blue_candidate.response
                ),
            )
        )

        return (
            self.red_generator.generate_one(
                prompt=prompt,

                temperature=(
                    config.CHALLENGE_TEMPERATURE
                ),

                generation_index=(
                    blue_candidate.candidate_index
                ),

                metadata={
                    "turn": 3,

                    "generation_type": (
                        "red_challenge"
                    ),

                    "defense_candidate_index": (
                        blue_candidate.candidate_index
                    ),
                },

                turn=3,

                generation_type="red_challenge",
            )
        )

    # ==================================================================
    # INTERACTION
    # ==================================================================

    def _evaluate_interaction(
        self,
        scenario: Scenario,
        blue_candidate: ScoredCandidate,
        challenge: GenerationResult,
    ) -> Dict[str, Any]:

        result = (
            self.oracle.evaluate_interaction(
                response=challenge.response,

                scenario=scenario,

                blue_defense=(
                    blue_candidate.response
                ),
            )
        )

        if hasattr(
            result,
            "to_dict",
        ):

            interaction = (
                result.to_dict()
            )

        elif isinstance(
            result,
            dict,
        ):

            interaction = dict(
                result
            )

        else:

            interaction = dict(
                result.__dict__
            )

        interaction[
            "candidate_index"
        ] = (
            blue_candidate.candidate_index
        )

        interaction[
            "defense_candidate_index"
        ] = (
            blue_candidate.candidate_index
        )

        interaction[
            "red_response"
        ] = challenge.response

        interaction[
            "blue_defense"
        ] = blue_candidate.response

        interaction[
            "challenge_prompt"
        ] = challenge.prompt

        return interaction

    # ==================================================================
    # INTERACTION REWARDS
    # ==================================================================

    def _apply_interaction_rewards(
        self,
        all_blue_candidates: List[ScoredCandidate],
        top_k_candidates: List[ScoredCandidate],
        interaction_results: List[Dict[str, Any]],
    ) -> None:

        rewards_by_index: Dict[
            int,
            float,
        ] = {}

        # --------------------------------------------------------------
        # Collect rewards ONLY from actual Turn-3 evaluations.
        # --------------------------------------------------------------

        for result in interaction_results:

            index = result.get(
                "candidate_index"
            )

            if index is None:

                index = result.get(
                    "defense_candidate_index"
                )

            if index is None:

                continue

            if "blue_reward" not in result:

                raise KeyError(
                    "Interaction result is missing "
                    "'blue_reward'."
                )

            rewards_by_index[
                int(index)
            ] = float(
                result["blue_reward"]
            )

        # --------------------------------------------------------------
        # These are the only candidates that were actually tested.
        # --------------------------------------------------------------

        tested_indices = {
            candidate.candidate_index
            for candidate in top_k_candidates
        }

        # --------------------------------------------------------------
        # IMPORTANT SOGARL RULE:
        #
        # Tested Top-K candidates:
        #     interaction_reward = actual Turn-3 reward
        #
        # Untested candidates:
        #     interaction_reward = None
        #
        # None is required here because RewardManager must EXCLUDE
        # untested candidates from interaction normalization.
        #
        # A zero reward would incorrectly turn "not tested" into an
        # observed zero-reward interaction.
        # --------------------------------------------------------------

        for candidate in all_blue_candidates:

            if (
                candidate.candidate_index
                not in tested_indices
            ):

                candidate.interaction_reward = None

            else:

                if (
                    candidate.candidate_index
                    not in rewards_by_index
                ):

                    raise ValueError(
                        "Top-K Blue candidate was selected "
                        "for interaction testing but no "
                        "interaction reward was returned."
                    )

                candidate.interaction_reward = (
                    rewards_by_index[
                        candidate.candidate_index
                    ]
                )

    # ==================================================================
    # REFERENCE GENERATOR
    # ==================================================================

    def _get_reference_generator(
        self,
        generator: Generator,
        trainer: GRPOTrainer,
    ) -> Generator:

        role = generator.get_role()

        if role in self._reference_generators:

            return (
                self._reference_generators[
                    role
                ]
            )

        reference_model = (
            trainer.get_reference_model()
        )

        generation_config = (
            generator.get_generation_config()
        )

        reference_generator = Generator(
            model=reference_model,

            tokenizer=(
                generator.get_tokenizer()
            ),

            device=(
                generator.get_device()
            ),

            role=role,

            model_name=(
                generation_config[
                    "model_name"
                ]
            ),

            max_input_tokens=(
                generation_config[
                    "max_input_tokens"
                ]
            ),

            max_new_tokens=(
                generation_config[
                    "max_new_tokens"
                ]
            ),

            top_p=(
                generation_config[
                    "top_p"
                ]
            ),

            do_sample=False,

            generation_batch_size=(
                generation_config[
                    "generation_batch_size"
                ]
            ),

            logger=None,
        )

        reference_generator.eval()

        self._reference_generators[
            role
        ] = reference_generator

        return reference_generator

    # ==================================================================
    # TRAINING LOG PROBABILITIES
    # ==================================================================

    def _prepare_training_log_probs(
        self,
        generator: Generator,
        candidates: List[ScoredCandidate],
        trainer: Optional[GRPOTrainer],
    ) -> None:

        if not candidates:

            return

        if trainer is None:

            return

        prompts = [
            candidate.prompt
            for candidate in candidates
        ]

        responses = [
            candidate.response
            for candidate in candidates
        ]

        was_training = (
            generator.is_training()
        )

        generator.eval()

        try:

            # ----------------------------------------------------------
            # Old rollout policy
            # ----------------------------------------------------------

            old_log_probs = (
                generator.compute_log_probs(
                    prompts=prompts,
                    responses=responses,
                    require_grad=False,
                )
            )

            # ----------------------------------------------------------
            # Frozen reference policy
            #
            # Keep all no-grad scoring ahead of the differentiable
            # current-policy scoring. Otherwise the autograd graphs for
            # all current candidates remain resident while the reference
            # model is scored, unnecessarily increasing peak VRAM.
            # ----------------------------------------------------------

            reference_generator = (
                self._get_reference_generator(
                    generator=generator,
                    trainer=trainer,
                )
            )

            reference_log_probs = (
                reference_generator
                .compute_log_probs(
                    prompts=prompts,
                    responses=responses,
                    require_grad=False,
                )
            )

            # ----------------------------------------------------------
            # Current trainable policy
            #
            # Do this last because these tensors retain autograd graphs
            # until GRPO backward().
            # ----------------------------------------------------------

            current_log_probs = (
                generator.compute_log_probs(
                    prompts=prompts,
                    responses=responses,
                    require_grad=True,
                )
            )

        finally:

            if was_training:

                generator.train()

            else:

                generator.eval()

        if not (
            len(old_log_probs)
            == len(current_log_probs)
            == len(reference_log_probs)
            == len(candidates)
        ):

            raise ValueError(
                "Log-probability output count does not "
                "match candidate count."
            )

        for (
            candidate,
            current,
            old,
            reference,
        ) in zip(
            candidates,
            current_log_probs,
            old_log_probs,
            reference_log_probs,
        ):

            candidate.metadata[
                "current_log_probs"
            ] = current

            candidate.metadata[
                "old_log_probs"
            ] = old.detach()

            candidate.metadata[
                "reference_log_probs"
            ] = reference.detach()

    # ==================================================================
    # RED UPDATE
    # ==================================================================

    def _update_red(
        self,
        candidates: List[ScoredCandidate],
    ) -> Optional[Dict[str, float]]:

        return self._update_policy(
            trainer=self.red_trainer,

            candidates=candidates,

            advantage_key="oracle_advantage",
        )

    # ==================================================================
    # BLUE UPDATE
    # ==================================================================

    def _update_blue(
        self,
        candidates: List[ScoredCandidate],
    ) -> Optional[Dict[str, float]]:

        return self._update_policy(
            trainer=self.blue_trainer,

            candidates=candidates,

            advantage_key="final_advantage",
        )

    # ==================================================================
    # GRPO UPDATE
    # ==================================================================

    def _update_policy(
        self,
        trainer: Optional[GRPOTrainer],
        candidates: List[ScoredCandidate],
        advantage_key: str,
    ) -> Optional[Dict[str, float]]:

        if trainer is None:

            return None

        valid_candidates = []

        for candidate in candidates:

            advantage = getattr(
                candidate,
                advantage_key,
                None,
            )

            if advantage is None:

                continue

            required = (
                "current_log_probs",
                "old_log_probs",
                "reference_log_probs",
            )

            if not all(
                key in candidate.metadata
                for key in required
            ):

                continue

            valid_candidates.append(
                candidate
            )

        if not valid_candidates:

            return None

        current = [
            candidate.metadata[
                "current_log_probs"
            ]
            for candidate in valid_candidates
        ]

        old = [
            candidate.metadata[
                "old_log_probs"
            ]
            for candidate in valid_candidates
        ]

        reference = [
            candidate.metadata[
                "reference_log_probs"
            ]
            for candidate in valid_candidates
        ]

        current_tensor, mask = (
            self._pad_log_probs(
                current
            )
        )

        old_tensor, _ = (
            self._pad_log_probs(
                old,
                device=(
                    current_tensor.device
                ),
            )
        )

        reference_tensor, _ = (
            self._pad_log_probs(
                reference,
                device=(
                    current_tensor.device
                ),
            )
        )

        advantages = torch.tensor(
            [
                float(
                    getattr(
                        candidate,
                        advantage_key,
                    )
                )
                for candidate
                in valid_candidates
            ],
            dtype=torch.float32,
            device=current_tensor.device,
        )

        try:

            metrics = trainer.update(
                current_log_probs=current_tensor,

                old_log_probs=old_tensor,

                reference_log_probs=(
                    reference_tensor
                ),

                advantages=advantages,

                normalize_advantages=True,

                response_mask=mask,
            )

        finally:

            # ----------------------------------------------------------
            # current_log_probs retain the autograd graph. Always remove
            # them from candidate metadata, including when backward or
            # optimizer code raises, so failed updates do not keep stale
            # graphs alive.
            # ----------------------------------------------------------

            self._clear_training_tensors(
                candidates
            )

        return metrics

    # ==================================================================
    # LOG-PROBABILITY PADDING
    # ==================================================================

    @staticmethod
    def _pad_log_probs(
        values: Sequence[Any],
        device: Optional[
            torch.device
        ] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if not values:

            raise ValueError(
                "Cannot pad empty log-probability values."
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

        target_device = (
            tensors[0].device
        )

        target_dtype = (
            tensors[0].dtype
        )

        result = torch.zeros(
            (
                len(tensors),
                max_length,
            ),
            dtype=target_dtype,
            device=target_device,
        )

        mask = torch.zeros(
            (
                len(tensors),
                max_length,
            ),
            dtype=torch.bool,
            device=target_device,
        )

        for index, tensor in enumerate(
            tensors
        ):

            length = tensor.numel()

            result[
                index,
                :length,
            ] = tensor

            mask[
                index,
                :length,
            ] = True

        return result, mask

    @staticmethod
    def _clear_training_tensors(
        candidates: List[ScoredCandidate],
    ) -> None:

        for candidate in candidates:

            for key in (
                "current_log_probs",
                "old_log_probs",
                "reference_log_probs",
            ):

                candidate.metadata.pop(
                    key,
                    None,
                )

    # ==================================================================
    # METRICS
    # ==================================================================

    def _build_episode_metrics(
        self,
        red_candidates: List[ScoredCandidate],
        blue_candidates: List[ScoredCandidate],
        top_k_blue: List[ScoredCandidate],
        interaction_results: List[Dict[str, Any]],
        red_update: Optional[Dict[str, float]],
        blue_update: Optional[Dict[str, float]],
        confidence_gate_passed: bool = False,
    ) -> Dict[str, Any]:

        red_rewards = [
            candidate.oracle_reward
            for candidate in red_candidates
        ]

        blue_rewards = [
            candidate.oracle_reward
            for candidate in blue_candidates
        ]

        interaction_rewards = [
            candidate.interaction_reward
            for candidate in blue_candidates
            if candidate.interaction_reward
            is not None
        ]

        red_advantages = [
            candidate.oracle_advantage
            for candidate in red_candidates
            if candidate.oracle_advantage
            is not None
        ]

        blue_oracle_advantages = [
            candidate.oracle_advantage
            for candidate in blue_candidates
            if candidate.oracle_advantage
            is not None
        ]

        blue_interaction_advantages = [
            candidate.interaction_advantage
            for candidate in blue_candidates
            if candidate.interaction_advantage
            is not None
        ]

        blue_final_advantages = [
            candidate.final_advantage
            for candidate in blue_candidates
            if candidate.final_advantage
            is not None
        ]

        interaction_cases = {}

        for result in interaction_results:

            case_id = result.get(
                "case_id",
                result.get(
                    "case"
                ),
            )

            if case_id is None:

                continue

            key = str(
                case_id
            )

            interaction_cases[key] = (
                interaction_cases.get(
                    key,
                    0,
                )
                + 1
            )

        # Keep the existing nested metrics for backward compatibility,
        # while also exposing the important scalar values directly so
        # training_metrics.jsonl and the Red/Blue CSVs are easy to analyze.
        red_update = red_update or {}
        blue_update = blue_update or {}

        metrics = {
            "red_rollouts": len(
                red_candidates
            ),

            "blue_rollouts": len(
                blue_candidates
            ),

            "top_k": len(
                top_k_blue
            ),

            "interaction_tests": len(
                interaction_results
            ),

            "red_mean_oracle_reward": (
                self._mean(
                    red_rewards
                )
            ),

            "red_best_oracle_reward": (
                max(red_rewards)
                if red_rewards
                else 0.0
            ),

            "red_std_oracle_reward": (
                self._std(
                    red_rewards
                )
            ),

            "blue_mean_oracle_reward": (
                self._mean(
                    blue_rewards
                )
            ),

            "blue_best_oracle_reward": (
                max(blue_rewards)
                if blue_rewards
                else 0.0
            ),

            "blue_std_oracle_reward": (
                self._std(
                    blue_rewards
                )
            ),

            "blue_mean_interaction_reward": (
                self._mean(
                    interaction_rewards
                )
            ),

            "red_mean_oracle_advantage": (
                self._mean(
                    red_advantages
                )
            ),

            "blue_mean_oracle_advantage": (
                self._mean(
                    blue_oracle_advantages
                )
            ),

            "blue_mean_interaction_advantage": (
                self._mean(
                    blue_interaction_advantages
                )
            ),

            "blue_mean_final_advantage": (
                self._mean(
                    blue_final_advantages
                )
            ),

            "confidence_gate_passed": bool(
                confidence_gate_passed
            ),

            "interaction_case_1_count": int(
                interaction_cases.get("1", 0)
            ),

            "interaction_case_2_count": int(
                interaction_cases.get("2", 0)
            ),

            "interaction_case_3_count": int(
                interaction_cases.get("3", 0)
            ),

            "interaction_case_4_count": int(
                interaction_cases.get("4", 0)
            ),

            "interaction_case_5_count": int(
                interaction_cases.get("5", 0)
            ),

            "interaction_cases": (
                interaction_cases
            ),

            # Preserve the existing nested update dictionaries.
            "red_update": red_update,

            "blue_update": blue_update,
        }

        # GRPOTrainer.update() already returns scalar metrics. Expose them
        # under stable role-specific names without changing the trainer API.
        metrics.update({
            "red_grpo_total_loss": float(
                red_update.get(
                    "total_loss",
                    red_update.get("loss", 0.0),
                )
            ),
            "red_grpo_policy_loss": float(
                red_update.get("policy_loss", 0.0)
            ),
            "red_grpo_kl_loss": float(
                red_update.get("kl_loss", 0.0)
            ),
            "red_grpo_mean_advantage": float(
                red_update.get("mean_advantage", 0.0)
            ),
            "red_grpo_mean_ratio": float(
                red_update.get("mean_ratio", 0.0)
            ),
            "red_grpo_clipped_fraction": float(
                red_update.get("clipped_fraction", 0.0)
            ),
            "red_grpo_valid_token_fraction": float(
                red_update.get("valid_token_fraction", 0.0)
            ),
            "blue_grpo_total_loss": float(
                blue_update.get(
                    "total_loss",
                    blue_update.get("loss", 0.0),
                )
            ),
            "blue_grpo_policy_loss": float(
                blue_update.get("policy_loss", 0.0)
            ),
            "blue_grpo_kl_loss": float(
                blue_update.get("kl_loss", 0.0)
            ),
            "blue_grpo_mean_advantage": float(
                blue_update.get("mean_advantage", 0.0)
            ),
            "blue_grpo_mean_ratio": float(
                blue_update.get("mean_ratio", 0.0)
            ),
            "blue_grpo_clipped_fraction": float(
                blue_update.get("clipped_fraction", 0.0)
            ),
            "blue_grpo_valid_token_fraction": float(
                blue_update.get("valid_token_fraction", 0.0)
            ),
        })

        return metrics

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:

        if not values:

            return 0.0

        return float(
            sum(values)
            / len(values)
        )

    @staticmethod
    def _std(
        values: Sequence[float],
    ) -> float:

        if not values:

            return 0.0

        mean = sum(values) / len(values)

        return float(
            (
                sum(
                    (value - mean) ** 2
                    for value in values
                )
                / len(values)
            ) ** 0.5
        )

    # ==================================================================
    # REPLAY + METRICS
    # ==================================================================

    def _log_episode(
        self,
        result: EpisodeResult,
    ) -> None:

        if self.replay_buffer is not None:

            record = ReplayRecord(
                scenario_id=(
                    result.scenario_id
                ),

                episode_id=(
                    result.episode_id
                ),

                epoch=result.epoch,

                category=(
                    result.red_candidates[0].metadata.get(
                        "scenario_category",
                        "unknown",
                    )
                    if result.red_candidates
                    else "unknown"
                ),

                red_attack_candidates=[
                    EpisodeResult._candidate_to_dict(
                        candidate
                    )
                    for candidate
                    in result.red_candidates
                ],

                blue_defense_candidates=[
                    EpisodeResult._candidate_to_dict(
                        candidate
                    )
                    for candidate
                    in result.blue_candidates
                ],

                top_k_blue_candidates=[
                    EpisodeResult._candidate_to_dict(
                        candidate
                    )
                    for candidate
                    in result.top_k_blue_candidates
                ],

                episode_metrics=(
                    result.episode_metrics
                ),

                interaction_results=(
                    result.interaction_results
                ),

                metadata={
                    "epoch": result.epoch,

                    "episode_metrics": (
                        result.episode_metrics
                    ),

                    "top_k_candidate_indices": [
                        candidate.candidate_index
                        for candidate
                        in result.top_k_blue_candidates
                    ],
                },
            )

            self.replay_buffer.add(
                record
            )

        if self.metrics_logger is not None:

            base_metrics = dict(
                result.episode_metrics
            )

            self.metrics_logger.log_episode(
                episode_id=(
                    result.episode_id
                ),

                scenario_id=(
                    result.scenario_id
                ),

                epoch=result.epoch,

                metrics=base_metrics,
            )

            # Dedicated Red row. Existing MetricsLogger API is preserved.
            red_metrics = {
                "episode_id": result.episode_id,
                "scenario_id": result.scenario_id,
                "epoch": result.epoch,
                "rollouts": base_metrics[
                    "red_rollouts"
                ],
                "mean_oracle_reward": base_metrics[
                    "red_mean_oracle_reward"
                ],
                "best_oracle_reward": base_metrics[
                    "red_best_oracle_reward"
                ],
                "std_oracle_reward": base_metrics[
                    "red_std_oracle_reward"
                ],
                "mean_oracle_advantage": base_metrics[
                    "red_mean_oracle_advantage"
                ],
                "confidence_gate_passed": base_metrics[
                    "confidence_gate_passed"
                ],
                "grpo_total_loss": base_metrics[
                    "red_grpo_total_loss"
                ],
                "grpo_policy_loss": base_metrics[
                    "red_grpo_policy_loss"
                ],
                "grpo_kl_loss": base_metrics[
                    "red_grpo_kl_loss"
                ],
                "grpo_mean_advantage": base_metrics[
                    "red_grpo_mean_advantage"
                ],
                "grpo_mean_ratio": base_metrics[
                    "red_grpo_mean_ratio"
                ],
                "grpo_clipped_fraction": base_metrics[
                    "red_grpo_clipped_fraction"
                ],
                "grpo_valid_token_fraction": base_metrics[
                    "red_grpo_valid_token_fraction"
                ],
            }

            self.metrics_logger.log_red(
                red_metrics
            )

            # Dedicated Blue row.
            blue_metrics = {
                "episode_id": result.episode_id,
                "scenario_id": result.scenario_id,
                "epoch": result.epoch,
                "rollouts": base_metrics[
                    "blue_rollouts"
                ],
                "top_k": base_metrics[
                    "top_k"
                ],
                "interaction_tests": base_metrics[
                    "interaction_tests"
                ],
                "mean_oracle_reward": base_metrics[
                    "blue_mean_oracle_reward"
                ],
                "best_oracle_reward": base_metrics[
                    "blue_best_oracle_reward"
                ],
                "std_oracle_reward": base_metrics[
                    "blue_std_oracle_reward"
                ],
                "mean_oracle_advantage": base_metrics[
                    "blue_mean_oracle_advantage"
                ],
                "mean_interaction_reward": base_metrics[
                    "blue_mean_interaction_reward"
                ],
                "mean_interaction_advantage": base_metrics[
                    "blue_mean_interaction_advantage"
                ],
                "mean_final_advantage": base_metrics[
                    "blue_mean_final_advantage"
                ],
                "interaction_case_1_count": base_metrics[
                    "interaction_case_1_count"
                ],
                "interaction_case_2_count": base_metrics[
                    "interaction_case_2_count"
                ],
                "interaction_case_3_count": base_metrics[
                    "interaction_case_3_count"
                ],
                "interaction_case_4_count": base_metrics[
                    "interaction_case_4_count"
                ],
                "interaction_case_5_count": base_metrics[
                    "interaction_case_5_count"
                ],
                "grpo_total_loss": base_metrics[
                    "blue_grpo_total_loss"
                ],
                "grpo_policy_loss": base_metrics[
                    "blue_grpo_policy_loss"
                ],
                "grpo_kl_loss": base_metrics[
                    "blue_grpo_kl_loss"
                ],
                "grpo_mean_advantage": base_metrics[
                    "blue_grpo_mean_advantage"
                ],
                "grpo_mean_ratio": base_metrics[
                    "blue_grpo_mean_ratio"
                ],
                "grpo_clipped_fraction": base_metrics[
                    "blue_grpo_clipped_fraction"
                ],
                "grpo_valid_token_fraction": base_metrics[
                    "blue_grpo_valid_token_fraction"
                ],
            }

            self.metrics_logger.log_blue(
                blue_metrics
            )

        if self.logger:

            metrics = (
                result.episode_metrics
            )

            self.logger.info(
                f"Episode {result.episode_id} | "
                f"Red="
                f"{metrics['red_mean_oracle_reward']:.3f} | "
                f"Blue="
                f"{metrics['blue_mean_oracle_reward']:.3f} | "
                f"Interaction="
                f"{metrics['blue_mean_interaction_reward']:.3f}"
            )