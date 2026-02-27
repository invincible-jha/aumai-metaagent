"""aumai-metaagent quickstart example.

Demonstrates four common usage patterns for the aumai-metaagent library.
Run directly to verify your installation:

    python examples/quickstart.py

All demos are self-contained. No real training environment is required —
the built-in synthetic fitness function is used unless you supply your own.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from aumai_metaagent.core import BlueprintGenerator, EvolutionEngine
from aumai_metaagent.models import (
    ActivationFn,
    AgentBlueprint,
    EvolutionConfig,
    OptimizerType,
    PerformanceMetric,
)


# ---------------------------------------------------------------------------
# Demo 1: Generate and inspect random blueprints
# ---------------------------------------------------------------------------


def demo_generate_blueprints() -> None:
    """Generate a small population of random AgentBlueprints.

    BlueprintGenerator samples uniformly from the architecture search space.
    Seeding the RNG makes results reproducible.
    """
    print("=== Demo 1: Generate Random Blueprints ===")

    rng = random.Random(42)
    generator = BlueprintGenerator(rng=rng)

    population = [generator.random(generation=0) for _ in range(5)]

    for blueprint in population:
        param_count = blueprint.parameter_count()
        print(
            f"  {blueprint.blueprint_id}"
            f"  layers={blueprint.hidden_layers}"
            f"  act={blueprint.activation.value:<8s}"
            f"  opt={blueprint.optimizer.value:<8s}"
            f"  lr={blueprint.learning_rate:.2e}"
            f"  params={param_count}"
        )
    print()


# ---------------------------------------------------------------------------
# Demo 2: Mutate and cross over blueprints
# ---------------------------------------------------------------------------


def demo_mutation_and_crossover() -> None:
    """Show how BlueprintGenerator transforms blueprints.

    Mutation applies random perturbations to each hyperparameter independently.
    Crossover mixes two parents using uniform crossover — each hyperparameter
    is independently drawn from either parent with 50% probability.
    """
    print("=== Demo 2: Mutation and Crossover ===")

    rng = random.Random(7)
    generator = BlueprintGenerator(rng=rng)

    parent_a = generator.random(generation=0)
    parent_b = generator.random(generation=0)

    print(f"Parent A: {parent_a.blueprint_id}  layers={parent_a.hidden_layers}")
    print(f"Parent B: {parent_b.blueprint_id}  layers={parent_b.hidden_layers}")

    # Mutate parent A — each hyperparameter has a 30% chance of changing.
    mutant = generator.mutate(parent_a, mutation_rate=0.3)
    print(f"Mutant:   {mutant.blueprint_id}  layers={mutant.hidden_layers}"
          f"  parents={mutant.parent_ids}")

    # Cross over both parents.
    child = generator.crossover(parent_a, parent_b)
    print(f"Child:    {child.blueprint_id}  layers={child.hidden_layers}"
          f"  parents={child.parent_ids}")
    print()


# ---------------------------------------------------------------------------
# Demo 3: Run evolution with the built-in fitness function
# ---------------------------------------------------------------------------


def demo_evolution_default_fitness() -> None:
    """Run the evolutionary loop using the built-in synthetic fitness proxy.

    The default fitness function rewards moderate-depth networks with tanh
    activation and Adam optimizer — useful for smoke-testing the engine.
    Replace it with a real environment evaluator for production use.
    """
    print("=== Demo 3: Evolution with Default Fitness Function ===")

    config = EvolutionConfig(
        population_size=20,
        generations=10,
        tournament_size=4,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_fraction=0.1,
        seed=42,
    )

    engine = EvolutionEngine(config)
    best = engine.evolve()

    print(f"Best blueprint ID: {best.blueprint_id}")
    print(f"Fitness:           {best.fitness:.4f}")
    print(f"Episode reward:    {best.episode_reward:.2f}")
    print(f"Success rate:      {best.success_rate:.1%}")
    print(f"Found at gen:      {best.generation}")

    # Show average fitness per generation from history.
    history = engine.get_history()
    by_gen: dict[int, list[float]] = defaultdict(list)
    for metric in history:
        by_gen[metric.generation].append(metric.fitness)

    print("\nAverage fitness per generation:")
    for gen_index in sorted(by_gen):
        fitnesses = by_gen[gen_index]
        avg = sum(fitnesses) / len(fitnesses)
        bar = "#" * int(avg * 30)
        print(f"  Gen {gen_index:2d}  avg={avg:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Evolution with a custom fitness function
# ---------------------------------------------------------------------------


def demo_evolution_custom_fitness() -> None:
    """Inject a custom fitness function into EvolutionEngine.

    In production, your fitness function would spin up a real environment,
    run evaluation episodes, and return a scalar score. Here we use a simple
    heuristic that rewards compact architectures with high exploration rates,
    demonstrating how to wire up any Callable[[AgentBlueprint], float].
    """
    print("=== Demo 4: Evolution with Custom Fitness Function ===")

    def compact_explorer_fitness(blueprint: AgentBlueprint) -> float:
        """Reward small, exploration-focused architectures.

        Prefers:
          - Fewer total parameters (compact models)
          - Higher exploration_rate
          - Lower dropout_rate
        """
        total_neurons = sum(blueprint.hidden_layers)
        compactness = 1.0 / (1.0 + total_neurons / 256.0)
        exploration_bonus = blueprint.exploration_rate
        dropout_penalty = blueprint.dropout_rate * 0.3
        return max(0.0, min(1.0, compactness * 0.5 + exploration_bonus * 0.5 - dropout_penalty))

    config = EvolutionConfig(
        population_size=15,
        generations=8,
        tournament_size=3,
        mutation_rate=0.25,
        crossover_rate=0.6,
        elite_fraction=0.15,
        seed=99,
    )

    engine = EvolutionEngine(config, fitness_fn=compact_explorer_fitness)
    best = engine.evolve()

    print(f"Best blueprint:    {best.blueprint_id}")
    print(f"Custom fitness:    {best.fitness:.4f}")
    print(f"Generation found:  {best.generation}")

    # Retrieve the actual blueprint from the last generation via history.
    # (PerformanceMetric stores blueprint_id, not the blueprint itself.)
    rng = random.Random(config.seed)
    generator = BlueprintGenerator(rng=rng)
    example_winner = generator.random(generation=best.generation)
    print(f"Example compact architecture: layers={example_winner.hidden_layers}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all aumai-metaagent quickstart demos."""
    print("aumai-metaagent quickstart\n")
    demo_generate_blueprints()
    demo_mutation_and_crossover()
    demo_evolution_default_fitness()
    demo_evolution_custom_fitness()
    print("All demos complete.")


if __name__ == "__main__":
    main()
