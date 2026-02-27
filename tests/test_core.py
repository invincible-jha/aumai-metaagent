"""Comprehensive tests for aumai_metaagent core and models.

Covers:
- AgentBlueprint model validation and parameter_count
- ActivationFn and OptimizerType enums
- PerformanceMetric model
- EvolutionConfig model
- BlueprintGenerator: random, mutate, crossover
- EvolutionEngine: evolve, get_history, tournament selection
- Reproducibility with seed
- Fitness function correctness
- Edge cases
"""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from aumai_metaagent.core import BlueprintGenerator, EvolutionEngine
from aumai_metaagent.models import (
    ActivationFn,
    AgentBlueprint,
    EvolutionConfig,
    OptimizerType,
    PerformanceMetric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture()
def generator(rng: random.Random) -> BlueprintGenerator:
    return BlueprintGenerator(rng=rng)


@pytest.fixture()
def small_config() -> EvolutionConfig:
    return EvolutionConfig(population_size=6, generations=3, seed=42)


@pytest.fixture()
def engine(small_config: EvolutionConfig) -> EvolutionEngine:
    return EvolutionEngine(small_config)


@pytest.fixture()
def sample_blueprint() -> AgentBlueprint:
    return AgentBlueprint(
        blueprint_id="bp_test",
        hidden_layers=[64, 128],
        activation=ActivationFn.RELU,
        optimizer=OptimizerType.ADAM,
        learning_rate=1e-3,
        dropout_rate=0.2,
        memory_size=1000,
        exploration_rate=0.1,
    )


# ---------------------------------------------------------------------------
# ActivationFn and OptimizerType enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_activation_fn_values(self) -> None:
        assert ActivationFn.RELU == "relu"
        assert ActivationFn.TANH == "tanh"
        assert ActivationFn.SIGMOID == "sigmoid"
        assert ActivationFn.LINEAR == "linear"
        assert ActivationFn.GELU == "gelu"

    def test_optimizer_type_values(self) -> None:
        assert OptimizerType.ADAM == "adam"
        assert OptimizerType.SGD == "sgd"
        assert OptimizerType.RMSPROP == "rmsprop"
        assert OptimizerType.ADAMW == "adamw"

    def test_activation_fn_is_str(self) -> None:
        assert isinstance(ActivationFn.RELU, str)

    def test_optimizer_type_is_str(self) -> None:
        assert isinstance(OptimizerType.ADAM, str)

    def test_all_activation_functions_iterable(self) -> None:
        fns = list(ActivationFn)
        assert len(fns) == 5

    def test_all_optimizer_types_iterable(self) -> None:
        optimizers = list(OptimizerType)
        assert len(optimizers) == 4


# ---------------------------------------------------------------------------
# AgentBlueprint model tests
# ---------------------------------------------------------------------------


class TestAgentBlueprint:
    def test_blueprint_id_stored(self, sample_blueprint: AgentBlueprint) -> None:
        assert sample_blueprint.blueprint_id == "bp_test"

    def test_hidden_layers_stored(self, sample_blueprint: AgentBlueprint) -> None:
        assert sample_blueprint.hidden_layers == [64, 128]

    def test_activation_stored(self, sample_blueprint: AgentBlueprint) -> None:
        assert sample_blueprint.activation == ActivationFn.RELU

    def test_optimizer_stored(self, sample_blueprint: AgentBlueprint) -> None:
        assert sample_blueprint.optimizer == OptimizerType.ADAM

    def test_learning_rate_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", learning_rate=0.0)

    def test_negative_learning_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", learning_rate=-0.001)

    def test_dropout_rate_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", dropout_rate=-0.1)

    def test_dropout_rate_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", dropout_rate=0.91)

    def test_memory_size_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", memory_size=0)

    def test_exploration_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", exploration_rate=1.1)

    def test_exploration_rate_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", exploration_rate=-0.01)

    def test_generation_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentBlueprint(blueprint_id="bp1", generation=-1)

    def test_parent_ids_defaults_empty(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1")
        assert bp.parent_ids == []

    def test_parameter_count_single_layer(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1", hidden_layers=[64])
        assert bp.parameter_count() == 0  # Single layer has no layer-to-layer connections

    def test_parameter_count_two_layers(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1", hidden_layers=[64, 128])
        # prev=64, size=128: 64*128 + 128 = 8320
        assert bp.parameter_count() == 64 * 128 + 128

    def test_parameter_count_three_layers(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1", hidden_layers=[64, 128, 64])
        # 64->128: 64*128+128=8320, 128->64: 128*64+64=8256 -> total=16576
        expected = (64 * 128 + 128) + (128 * 64 + 64)
        assert bp.parameter_count() == expected

    def test_parameter_count_empty_layers(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1", hidden_layers=[])
        assert bp.parameter_count() == 0

    def test_extra_params_stored(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1", extra_params={"batch_size": 32})
        assert bp.extra_params["batch_size"] == 32

    def test_default_hidden_layers(self) -> None:
        bp = AgentBlueprint(blueprint_id="bp1")
        assert bp.hidden_layers == [64, 64]


# ---------------------------------------------------------------------------
# PerformanceMetric model tests
# ---------------------------------------------------------------------------


class TestPerformanceMetric:
    def test_blueprint_id_stored(self) -> None:
        m = PerformanceMetric(blueprint_id="bp1", fitness=0.8)
        assert m.blueprint_id == "bp1"

    def test_fitness_stored(self) -> None:
        m = PerformanceMetric(blueprint_id="bp1", fitness=0.75)
        assert m.fitness == 0.75

    def test_success_rate_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceMetric(blueprint_id="bp1", success_rate=-0.1)

    def test_success_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceMetric(blueprint_id="bp1", success_rate=1.1)

    def test_evaluation_steps_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceMetric(blueprint_id="bp1", evaluation_steps=-1)

    def test_notes_optional(self) -> None:
        m = PerformanceMetric(blueprint_id="bp1")
        assert m.notes is None

    def test_generation_stored(self) -> None:
        m = PerformanceMetric(blueprint_id="bp1", generation=5)
        assert m.generation == 5


# ---------------------------------------------------------------------------
# EvolutionConfig model tests
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self) -> None:
        config = EvolutionConfig()
        assert config.population_size == 20
        assert config.generations == 10
        assert config.tournament_size == 4
        assert config.mutation_rate == pytest.approx(0.2)
        assert config.crossover_rate == pytest.approx(0.7)
        assert config.elite_fraction == pytest.approx(0.1)
        assert config.seed is None

    def test_population_size_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(population_size=0)

    def test_generations_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(generations=0)

    def test_tournament_size_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(tournament_size=0)

    def test_mutation_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(mutation_rate=1.1)

    def test_crossover_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(crossover_rate=1.1)

    def test_elite_fraction_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionConfig(elite_fraction=1.1)

    def test_seed_stored(self) -> None:
        config = EvolutionConfig(seed=99)
        assert config.seed == 99


# ---------------------------------------------------------------------------
# BlueprintGenerator tests
# ---------------------------------------------------------------------------


class TestBlueprintGenerator:
    def test_random_returns_agent_blueprint(self, generator: BlueprintGenerator) -> None:
        bp = generator.random()
        assert isinstance(bp, AgentBlueprint)

    def test_random_blueprint_id_has_prefix(self, generator: BlueprintGenerator) -> None:
        bp = generator.random()
        assert bp.blueprint_id.startswith("bp_")

    def test_random_generation_stamped(self, generator: BlueprintGenerator) -> None:
        bp = generator.random(generation=3)
        assert bp.generation == 3

    def test_random_hidden_layers_in_valid_range(self, generator: BlueprintGenerator) -> None:
        for _ in range(10):
            bp = generator.random()
            assert 1 <= len(bp.hidden_layers) <= 4

    def test_random_activation_is_valid(self, generator: BlueprintGenerator) -> None:
        bp = generator.random()
        assert bp.activation in list(ActivationFn)

    def test_random_optimizer_is_valid(self, generator: BlueprintGenerator) -> None:
        bp = generator.random()
        assert bp.optimizer in list(OptimizerType)

    def test_random_learning_rate_positive(self, generator: BlueprintGenerator) -> None:
        bp = generator.random()
        assert bp.learning_rate > 0.0

    def test_random_dropout_rate_in_range(self, generator: BlueprintGenerator) -> None:
        for _ in range(10):
            bp = generator.random()
            assert 0.0 <= bp.dropout_rate <= 0.5

    def test_random_exploration_rate_in_range(self, generator: BlueprintGenerator) -> None:
        for _ in range(10):
            bp = generator.random()
            assert 0.01 <= bp.exploration_rate <= 0.5

    def test_mutate_returns_new_blueprint(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        mutated = generator.mutate(sample_blueprint)
        assert isinstance(mutated, AgentBlueprint)

    def test_mutate_does_not_modify_original(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        original_id = sample_blueprint.blueprint_id
        original_layers = list(sample_blueprint.hidden_layers)
        generator.mutate(sample_blueprint, mutation_rate=1.0)
        assert sample_blueprint.blueprint_id == original_id
        assert sample_blueprint.hidden_layers == original_layers

    def test_mutate_new_id_assigned(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        mutated = generator.mutate(sample_blueprint)
        assert mutated.blueprint_id != sample_blueprint.blueprint_id

    def test_mutate_preserves_generation(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        mutated = generator.mutate(sample_blueprint)
        assert mutated.generation == sample_blueprint.generation

    def test_mutate_parent_ids_include_original(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        mutated = generator.mutate(sample_blueprint)
        assert sample_blueprint.blueprint_id in mutated.parent_ids

    def test_mutate_zero_rate_keeps_params(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        # With rate=0, nothing mutates
        mutated = generator.mutate(sample_blueprint, mutation_rate=0.0)
        assert mutated.activation == sample_blueprint.activation
        assert mutated.optimizer == sample_blueprint.optimizer

    def test_crossover_returns_blueprint(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        bp_b = generator.random()
        child = generator.crossover(sample_blueprint, bp_b)
        assert isinstance(child, AgentBlueprint)

    def test_crossover_new_id(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        bp_b = generator.random()
        child = generator.crossover(sample_blueprint, bp_b)
        assert child.blueprint_id != sample_blueprint.blueprint_id
        assert child.blueprint_id != bp_b.blueprint_id

    def test_crossover_parent_ids_set(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        bp_b = generator.random()
        child = generator.crossover(sample_blueprint, bp_b)
        assert sample_blueprint.blueprint_id in child.parent_ids
        assert bp_b.blueprint_id in child.parent_ids

    def test_crossover_generation_is_max(
        self, generator: BlueprintGenerator
    ) -> None:
        bp_a = AgentBlueprint(blueprint_id="a", generation=3)
        bp_b = AgentBlueprint(blueprint_id="b", generation=7)
        child = generator.crossover(bp_a, bp_b)
        assert child.generation == 7

    def test_crossover_hidden_layers_non_empty(
        self, generator: BlueprintGenerator, sample_blueprint: AgentBlueprint
    ) -> None:
        bp_b = generator.random()
        child = generator.crossover(sample_blueprint, bp_b)
        assert len(child.hidden_layers) > 0

    def test_generator_no_rng_uses_global(self) -> None:
        gen = BlueprintGenerator()
        bp = gen.random()
        assert isinstance(bp, AgentBlueprint)

    def test_deterministic_with_same_seed(self) -> None:
        gen1 = BlueprintGenerator(rng=random.Random(99))
        gen2 = BlueprintGenerator(rng=random.Random(99))
        bp1 = gen1.random(generation=0)
        bp2 = gen2.random(generation=0)
        assert bp1.activation == bp2.activation
        assert bp1.optimizer == bp2.optimizer
        assert bp1.hidden_layers == bp2.hidden_layers


# ---------------------------------------------------------------------------
# EvolutionEngine tests
# ---------------------------------------------------------------------------


class TestEvolutionEngine:
    def test_evolve_returns_performance_metric(self, engine: EvolutionEngine) -> None:
        result = engine.evolve()
        assert isinstance(result, PerformanceMetric)

    def test_evolve_fitness_non_negative(self, engine: EvolutionEngine) -> None:
        result = engine.evolve()
        assert result.fitness >= 0.0

    def test_evolve_fitness_at_most_one(self, engine: EvolutionEngine) -> None:
        result = engine.evolve()
        assert result.fitness <= 1.0

    def test_history_populated_after_evolve(self, engine: EvolutionEngine) -> None:
        engine.evolve()
        history = engine.get_history()
        assert len(history) > 0

    def test_history_count_is_population_times_generations(
        self, small_config: EvolutionConfig
    ) -> None:
        engine = EvolutionEngine(small_config)
        engine.evolve()
        history = engine.get_history()
        expected = small_config.population_size * small_config.generations
        assert len(history) == expected

    def test_history_returns_copy(self, engine: EvolutionEngine) -> None:
        engine.evolve()
        h1 = engine.get_history()
        h2 = engine.get_history()
        assert h1 is not h2

    def test_deterministic_with_seed(self) -> None:
        config1 = EvolutionConfig(population_size=4, generations=2, seed=7)
        config2 = EvolutionConfig(population_size=4, generations=2, seed=7)
        result1 = EvolutionEngine(config1).evolve()
        result2 = EvolutionEngine(config2).evolve()
        assert result1.fitness == pytest.approx(result2.fitness)

    def test_custom_fitness_fn_used(self) -> None:
        config = EvolutionConfig(population_size=4, generations=2, seed=0)

        def always_one(bp: AgentBlueprint) -> float:
            return 1.0

        engine = EvolutionEngine(config, fitness_fn=always_one)
        result = engine.evolve()
        assert result.fitness == pytest.approx(1.0)

    def test_default_fitness_tanh_adam_bonus(self) -> None:
        bp = AgentBlueprint(
            blueprint_id="bp1",
            hidden_layers=[64, 64],  # depth_score=1.0
            activation=ActivationFn.TANH,
            optimizer=OptimizerType.ADAM,
            learning_rate=1e-3,
            dropout_rate=0.0,
        )
        fitness = EvolutionEngine._default_fitness(bp)
        assert fitness > 0.0
        assert fitness <= 1.0

    def test_default_fitness_non_optimal_lower(self) -> None:
        bp_opt = AgentBlueprint(
            blueprint_id="opt",
            hidden_layers=[64, 64],
            activation=ActivationFn.TANH,
            optimizer=OptimizerType.ADAM,
            learning_rate=1e-3,
            dropout_rate=0.0,
        )
        bp_bad = AgentBlueprint(
            blueprint_id="bad",
            hidden_layers=[64] * 4,  # too deep
            activation=ActivationFn.SIGMOID,
            optimizer=OptimizerType.SGD,
            learning_rate=1.0,  # too high
            dropout_rate=0.5,  # max penalty
        )
        fitness_opt = EvolutionEngine._default_fitness(bp_opt)
        fitness_bad = EvolutionEngine._default_fitness(bp_bad)
        assert fitness_opt > fitness_bad

    def test_evolve_best_has_blueprint_id(self, engine: EvolutionEngine) -> None:
        result = engine.evolve()
        assert result.blueprint_id != ""

    def test_evolve_with_single_generation(self) -> None:
        config = EvolutionConfig(population_size=5, generations=1, seed=0)
        engine = EvolutionEngine(config)
        result = engine.evolve()
        assert isinstance(result, PerformanceMetric)

    def test_evolve_with_high_mutation_rate(self) -> None:
        config = EvolutionConfig(population_size=4, generations=2, seed=1, mutation_rate=1.0)
        engine = EvolutionEngine(config)
        result = engine.evolve()
        assert result.fitness >= 0.0

    def test_evolve_with_no_crossover(self) -> None:
        config = EvolutionConfig(population_size=4, generations=2, seed=2, crossover_rate=0.0)
        engine = EvolutionEngine(config)
        result = engine.evolve()
        assert isinstance(result, PerformanceMetric)

    def test_tournament_select_returns_blueprint(self, engine: EvolutionEngine) -> None:
        population = [AgentBlueprint(blueprint_id=f"bp{i}") for i in range(5)]
        metrics = [PerformanceMetric(blueprint_id=f"bp{i}", fitness=float(i)) for i in range(5)]
        winner = engine._tournament_select(population, metrics)
        assert isinstance(winner, AgentBlueprint)

    def test_tournament_select_prefers_higher_fitness(self) -> None:
        config = EvolutionConfig(population_size=5, generations=1, seed=0, tournament_size=5)
        engine = EvolutionEngine(config)
        population = [AgentBlueprint(blueprint_id=f"bp{i}") for i in range(5)]
        metrics = [PerformanceMetric(blueprint_id=f"bp{i}", fitness=float(i)) for i in range(5)]
        # With tournament_size=5, all are contestants; best is bp4 (fitness=4)
        winner = engine._tournament_select(population, metrics)
        assert winner.blueprint_id == "bp4"

    def test_history_all_are_performance_metrics(self, engine: EvolutionEngine) -> None:
        engine.evolve()
        for metric in engine.get_history():
            assert isinstance(metric, PerformanceMetric)

    def test_evolve_success_rate_in_bounds(self, engine: EvolutionEngine) -> None:
        result = engine.evolve()
        assert 0.0 <= result.success_rate <= 1.0
