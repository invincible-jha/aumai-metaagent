"""Core evolutionary meta-learning engine.

Provides:
- BlueprintGenerator: random generation, mutation, and crossover of AgentBlueprints.
- EvolutionEngine: tournament selection, generational evolution loop.
"""

from __future__ import annotations

import random
import uuid
from typing import Callable, Optional

from .models import (
    ActivationFn,
    AgentBlueprint,
    EvolutionConfig,
    OptimizerType,
    PerformanceMetric,
)


# ---------------------------------------------------------------------------
# BlueprintGenerator
# ---------------------------------------------------------------------------


class BlueprintGenerator:
    """Generate, mutate, and combine AgentBlueprint objects.

    All randomness is seeded via the provided random.Random instance
    to ensure reproducibility.

    Example:
        >>> rng = random.Random(42)
        >>> gen = BlueprintGenerator(rng)
        >>> blueprint = gen.random(generation=0)
        >>> mutated = gen.mutate(blueprint, mutation_rate=0.3)
        >>> child = gen.crossover(blueprint, mutated)
    """

    _LAYER_SIZES = [16, 32, 64, 128, 256, 512]
    _MIN_LAYERS = 1
    _MAX_LAYERS = 4

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        """Initialise with an optional seeded RNG.

        Args:
            rng: A random.Random instance. Creates a new one if None.
        """
        self._rng = rng or random.Random()

    def random(self, generation: int = 0) -> AgentBlueprint:
        """Generate a fully random AgentBlueprint.

        Args:
            generation: Generation number to stamp on the blueprint.

        Returns:
            A randomly parameterised AgentBlueprint.
        """
        num_layers = self._rng.randint(self._MIN_LAYERS, self._MAX_LAYERS)
        hidden_layers = [self._rng.choice(self._LAYER_SIZES) for _ in range(num_layers)]

        return AgentBlueprint(
            blueprint_id=self._new_id(),
            hidden_layers=hidden_layers,
            activation=self._rng.choice(list(ActivationFn)),
            optimizer=self._rng.choice(list(OptimizerType)),
            learning_rate=10 ** self._rng.uniform(-5, -1),
            dropout_rate=round(self._rng.uniform(0.0, 0.5), 2),
            memory_size=self._rng.choice([256, 512, 1000, 2000, 5000]),
            exploration_rate=round(self._rng.uniform(0.01, 0.5), 3),
            generation=generation,
        )

    def mutate(self, blueprint: AgentBlueprint, mutation_rate: float = 0.2) -> AgentBlueprint:
        """Return a mutated copy of *blueprint*.

        Each hyperparameter is independently mutated with probability
        *mutation_rate*.  The original blueprint is not modified.

        Args:
            blueprint: The AgentBlueprint to mutate.
            mutation_rate: Probability of mutating each parameter.

        Returns:
            A new mutated AgentBlueprint.
        """
        layers = list(blueprint.hidden_layers)

        if self._rng.random() < mutation_rate:
            # Mutate architecture: add, remove, or resize a layer
            operation = self._rng.choice(["resize", "add", "remove"])
            if operation == "add" and len(layers) < self._MAX_LAYERS:
                layers.append(self._rng.choice(self._LAYER_SIZES))
            elif operation == "remove" and len(layers) > self._MIN_LAYERS:
                layers.pop(self._rng.randrange(len(layers)))
            elif layers:
                idx = self._rng.randrange(len(layers))
                layers[idx] = self._rng.choice(self._LAYER_SIZES)

        activation = (
            self._rng.choice(list(ActivationFn))
            if self._rng.random() < mutation_rate
            else blueprint.activation
        )
        optimizer = (
            self._rng.choice(list(OptimizerType))
            if self._rng.random() < mutation_rate
            else blueprint.optimizer
        )
        learning_rate = (
            10 ** self._rng.uniform(-5, -1)
            if self._rng.random() < mutation_rate
            else blueprint.learning_rate
        )
        dropout_rate = (
            round(self._rng.uniform(0.0, 0.5), 2)
            if self._rng.random() < mutation_rate
            else blueprint.dropout_rate
        )
        memory_size = (
            self._rng.choice([256, 512, 1000, 2000, 5000])
            if self._rng.random() < mutation_rate
            else blueprint.memory_size
        )
        exploration_rate = (
            round(self._rng.uniform(0.01, 0.5), 3)
            if self._rng.random() < mutation_rate
            else blueprint.exploration_rate
        )

        return AgentBlueprint(
            blueprint_id=self._new_id(),
            hidden_layers=layers,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            memory_size=memory_size,
            exploration_rate=exploration_rate,
            generation=blueprint.generation,
            parent_ids=[blueprint.blueprint_id],
        )

    def crossover(self, parent_a: AgentBlueprint, parent_b: AgentBlueprint) -> AgentBlueprint:
        """Create a child blueprint by crossing over two parents.

        Uses uniform crossover: each hyperparameter is independently
        drawn from either parent with 50% probability.

        Args:
            parent_a: First parent blueprint.
            parent_b: Second parent blueprint.

        Returns:
            A new child AgentBlueprint.
        """
        pick = self._rng.random

        # Crossover hidden layers: take longer or shorter randomly
        layers_a, layers_b = parent_a.hidden_layers, parent_b.hidden_layers
        min_len = min(len(layers_a), len(layers_b))
        max_len = max(len(layers_a), len(layers_b))
        target_len = self._rng.randint(min_len, max_len)
        longer = layers_a if len(layers_a) >= len(layers_b) else layers_b
        shorter = layers_a if len(layers_a) < len(layers_b) else layers_b
        # Pair up existing layers, draw remaining from longer parent
        child_layers = [
            layers_a[i] if pick() < 0.5 else layers_b[i]
            for i in range(min(min_len, target_len))
        ] + list(longer[min_len:target_len])

        return AgentBlueprint(
            blueprint_id=self._new_id(),
            hidden_layers=child_layers or [64],
            activation=parent_a.activation if pick() < 0.5 else parent_b.activation,
            optimizer=parent_a.optimizer if pick() < 0.5 else parent_b.optimizer,
            learning_rate=parent_a.learning_rate if pick() < 0.5 else parent_b.learning_rate,
            dropout_rate=parent_a.dropout_rate if pick() < 0.5 else parent_b.dropout_rate,
            memory_size=parent_a.memory_size if pick() < 0.5 else parent_b.memory_size,
            exploration_rate=parent_a.exploration_rate if pick() < 0.5 else parent_b.exploration_rate,
            generation=max(parent_a.generation, parent_b.generation),
            parent_ids=[parent_a.blueprint_id, parent_b.blueprint_id],
        )

    def _new_id(self) -> str:
        return f"bp_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# EvolutionEngine
# ---------------------------------------------------------------------------


class EvolutionEngine:
    """Run the evolutionary meta-learning loop.

    Uses tournament selection, elitism, and configurable crossover/mutation.
    The fitness function is injected at construction time — by default a simple
    synthetic function is used for demonstration.

    Example:
        >>> config = EvolutionConfig(population_size=10, generations=5, seed=42)
        >>> engine = EvolutionEngine(config)
        >>> best = engine.evolve()
        >>> print(best.blueprint_id, best.fitness)
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Optional[Callable[[AgentBlueprint], float]] = None,
    ) -> None:
        """Initialise the engine.

        Args:
            config: Evolution hyperparameters.
            fitness_fn: Optional custom fitness function. If None, uses a
                        synthetic proxy based on architecture shape.
        """
        self._config = config
        self._rng = random.Random(config.seed)
        self._generator = BlueprintGenerator(rng=self._rng)
        self._fitness_fn = fitness_fn or self._default_fitness
        self._history: list[PerformanceMetric] = []

    def evolve(self) -> PerformanceMetric:
        """Run the full evolution loop and return the best metric found.

        Returns:
            PerformanceMetric of the best blueprint found across all generations.
        """
        cfg = self._config
        population = [
            self._generator.random(generation=0) for _ in range(cfg.population_size)
        ]

        all_time_best: Optional[PerformanceMetric] = None

        for generation in range(cfg.generations):
            # Evaluate
            metrics: list[PerformanceMetric] = []
            for bp in population:
                fitness = self._fitness_fn(bp)
                metric = PerformanceMetric(
                    blueprint_id=bp.blueprint_id,
                    fitness=fitness,
                    episode_reward=fitness * 100,
                    success_rate=min(1.0, fitness),
                    generation=generation,
                )
                metrics.append(metric)
                self._history.append(metric)

            # Sort by fitness
            scored = sorted(
                zip(population, metrics), key=lambda pair: pair[1].fitness, reverse=True
            )

            # Track best
            if all_time_best is None or scored[0][1].fitness > all_time_best.fitness:
                all_time_best = scored[0][1]

            # Elites carry over
            elite_count = max(1, int(cfg.elite_fraction * cfg.population_size))
            new_population: list[AgentBlueprint] = [
                ind for ind, _ in scored[:elite_count]
            ]

            # Fill rest via tournament selection + crossover + mutation
            while len(new_population) < cfg.population_size:
                parent_a = self._tournament_select(population, metrics)
                if self._rng.random() < cfg.crossover_rate:
                    parent_b = self._tournament_select(population, metrics)
                    child = self._generator.crossover(parent_a, parent_b)
                else:
                    child = parent_a.model_copy(deep=True)

                if self._rng.random() < cfg.mutation_rate:
                    child = self._generator.mutate(child, mutation_rate=cfg.mutation_rate)

                child = child.model_copy(update={"generation": generation + 1})
                new_population.append(child)

            population = new_population

        return all_time_best or PerformanceMetric(blueprint_id="none", fitness=0.0)

    def get_history(self) -> list[PerformanceMetric]:
        """Return all PerformanceMetric records from the evolution run."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tournament_select(
        self, population: list[AgentBlueprint], metrics: list[PerformanceMetric]
    ) -> AgentBlueprint:
        """Select one individual via tournament selection.

        Args:
            population: Current generation blueprints.
            metrics: Corresponding performance metrics.

        Returns:
            Winner of the tournament.
        """
        fitness_map = {m.blueprint_id: m.fitness for m in metrics}
        k = min(self._config.tournament_size, len(population))
        contestants = self._rng.sample(population, k)
        winner = max(contestants, key=lambda bp: fitness_map.get(bp.blueprint_id, 0.0))
        return winner

    @staticmethod
    def _default_fitness(blueprint: AgentBlueprint) -> float:
        """Synthetic fitness proxy for demonstration.

        Rewards moderate-depth networks with tanh activation and Adam.
        Real-world use should inject a proper fitness function.

        Args:
            blueprint: Blueprint to evaluate.

        Returns:
            Fitness scalar in roughly [0, 1].
        """
        depth_score = 1.0 / (1.0 + abs(len(blueprint.hidden_layers) - 2))
        lr_score = 1.0 - abs(blueprint.learning_rate - 1e-3) / 1e-3
        lr_score = max(0.0, min(1.0, lr_score))
        activation_bonus = 0.1 if blueprint.activation == ActivationFn.TANH else 0.0
        optimizer_bonus = 0.05 if blueprint.optimizer == OptimizerType.ADAM else 0.0
        dropout_penalty = blueprint.dropout_rate * 0.2

        fitness = (
            depth_score * 0.4
            + lr_score * 0.3
            + activation_bonus
            + optimizer_bonus
            - dropout_penalty
        )
        return max(0.0, min(1.0, fitness))
