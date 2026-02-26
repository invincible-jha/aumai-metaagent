"""CLI entry point for aumai-metaagent.

Commands:
    evolve    -- run evolutionary meta-learning for N generations
    generate  -- generate a random agent blueprint
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .core import BlueprintGenerator, EvolutionEngine
from .models import EvolutionConfig


@click.group()
@click.version_option()
def main() -> None:
    """AumAI MetaAgent -- evolutionary meta-learning agent builder CLI."""


@main.command("evolve")
@click.option("--generations", default=10, show_default=True, type=int)
@click.option("--population", default=20, show_default=True, type=int)
@click.option("--tournament-size", default=4, show_default=True, type=int)
@click.option("--mutation-rate", default=0.2, show_default=True, type=float)
@click.option("--crossover-rate", default=0.7, show_default=True, type=float)
@click.option("--elite-fraction", default=0.1, show_default=True, type=float)
@click.option("--seed", default=None, type=int)
@click.option(
    "--output",
    "output_path",
    default="evolution_result.json",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--history-output", default=None, type=click.Path(path_type=Path))
def evolve_command(
    generations: int,
    population: int,
    tournament_size: int,
    mutation_rate: float,
    crossover_rate: float,
    elite_fraction: float,
    seed: int | None,
    output_path: Path,
    history_output: Path | None,
) -> None:
    """Run the evolutionary meta-learning loop.

    Example:

        aumai-metaagent evolve --generations 100 --population 30 --seed 42
    """
    config = EvolutionConfig(
        population_size=population,
        generations=generations,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_fraction=elite_fraction,
        seed=seed,
    )

    click.echo(
        f"Evolving {population} agents for {generations} generation(s) "
        f"(seed={seed}, tournament_size={tournament_size})"
    )

    engine = EvolutionEngine(config)
    best = engine.evolve()

    click.echo(f"\nBest blueprint: {best.blueprint_id}")
    click.echo(f"  Fitness       : {best.fitness:.4f}")
    click.echo(f"  Episode reward: {best.episode_reward:.2f}")
    click.echo(f"  Success rate  : {best.success_rate:.1%}")
    click.echo(f"  Generation    : {best.generation}")

    output_path.write_text(best.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"\nBest result saved to {output_path}")

    if history_output:
        history = engine.get_history()
        history_output.write_text(
            json.dumps([m.model_dump() for m in history], indent=2), encoding="utf-8"
        )
        click.echo(f"Evolution history saved to {history_output}")


@main.command("generate")
@click.option("--count", default=1, show_default=True, type=int, help="Number of blueprints to generate.")
@click.option("--seed", default=None, type=int)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output JSON path. Defaults to stdout.",
)
def generate_command(count: int, seed: int | None, output_path: Path | None) -> None:
    """Generate random agent blueprints.

    Example:

        aumai-metaagent generate --count 5 --seed 0
    """
    import random as _random
    rng = _random.Random(seed)
    generator = BlueprintGenerator(rng=rng)
    blueprints = [generator.random(generation=0) for _ in range(count)]

    output = json.dumps([bp.model_dump() for bp in blueprints], indent=2)

    if output_path:
        output_path.write_text(output, encoding="utf-8")
        click.echo(f"Generated {count} blueprint(s) -> {output_path}")
    else:
        click.echo(output)


if __name__ == "__main__":
    main()
