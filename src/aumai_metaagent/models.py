"""Pydantic v2 models for evolutionary meta-learning agent builder."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ActivationFn(str, Enum):
    """Supported activation functions for neural network layers."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    GELU = "gelu"


class OptimizerType(str, Enum):
    """Supported optimizers."""

    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class AgentBlueprint(BaseModel):
    """A parameterised description of an agent architecture.

    Attributes:
        blueprint_id: Unique identifier.
        hidden_layers: List of hidden layer sizes.
        activation: Activation function for hidden layers.
        optimizer: Optimizer type.
        learning_rate: Learning rate for the optimizer.
        dropout_rate: Dropout rate for regularisation.
        memory_size: Replay buffer / context window size.
        exploration_rate: Epsilon for epsilon-greedy exploration.
        generation: Evolution generation in which this was created.
        parent_ids: Blueprint IDs of parent(s) used to create this.
        extra_params: Extensible additional hyperparameters.
    """

    blueprint_id: str
    hidden_layers: list[int] = Field(default_factory=lambda: [64, 64])
    activation: ActivationFn = ActivationFn.RELU
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = Field(default=1e-3, gt=0.0)
    dropout_rate: float = Field(default=0.0, ge=0.0, le=0.9)
    memory_size: int = Field(default=1000, gt=0)
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    generation: int = Field(default=0, ge=0)
    parent_ids: list[str] = Field(default_factory=list)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def parameter_count(self) -> int:
        """Estimate total trainable parameter count from hidden_layers."""
        if not self.hidden_layers:
            return 0
        total = 0
        prev = self.hidden_layers[0]
        for size in self.hidden_layers[1:]:
            total += prev * size + size
            prev = size
        return total


class PerformanceMetric(BaseModel):
    """A measured performance outcome for an AgentBlueprint.

    Attributes:
        blueprint_id: Blueprint this metric belongs to.
        fitness: Primary fitness score (higher is better).
        episode_reward: Mean reward over evaluation episodes.
        success_rate: Fraction of tasks completed successfully.
        evaluation_steps: Number of environment steps used for evaluation.
        generation: Generation at which this was measured.
        notes: Optional contextual notes.
    """

    blueprint_id: str
    fitness: float = Field(default=0.0)
    episode_reward: float = Field(default=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation_steps: int = Field(default=0, ge=0)
    generation: int = Field(default=0, ge=0)
    notes: Optional[str] = None


class EvolutionConfig(BaseModel):
    """Hyperparameters for the evolutionary meta-learning loop.

    Attributes:
        population_size: Number of blueprints per generation.
        generations: Total number of evolution generations.
        tournament_size: Number of individuals in each tournament selection.
        mutation_rate: Probability of mutating each hyperparameter.
        crossover_rate: Probability of crossover vs. cloning.
        elite_fraction: Fraction of top individuals carried to next generation.
        seed: Optional random seed.
        evaluation_episodes: Episodes per fitness evaluation.
    """

    population_size: int = Field(default=20, gt=0)
    generations: int = Field(default=10, gt=0)
    tournament_size: int = Field(default=4, gt=0)
    mutation_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    elite_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: Optional[int] = None
    evaluation_episodes: int = Field(default=10, gt=0)
