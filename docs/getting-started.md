# Getting Started with aumai-metaagent

> **Experimental** — This is frontier research software. APIs may change between
> minor releases.

This guide walks you through installing `aumai-metaagent`, running your first
evolutionary architecture search, and understanding the common patterns for plugging
evolutionary meta-learning into your AI agent development workflow.

---

## Prerequisites

- Python 3.11 or later
- `pip` (or `uv` / `poetry`)
- Basic understanding of machine learning hyperparameters (learning rate, dropout, etc.)
- No ML framework required for the evolutionary loop itself — the fitness function you
  provide can use any framework (PyTorch, JAX, etc.)

---

## Installation

### From PyPI

```bash
pip install aumai-metaagent
```

### From source (development mode)

```bash
git clone https://github.com/aumai/aumai-metaagent
cd aumai-metaagent
pip install -e ".[dev]"
```

Verify your installation:

```bash
aumai-metaagent --version
# AumAI MetaAgent, version 0.1.0
```

Or in Python:

```python
import aumai_metaagent
print(aumai_metaagent.__version__)  # 0.1.0
```

---

## Step-by-Step Tutorial

### Step 1 — Understand AgentBlueprint

An `AgentBlueprint` is a complete hyperparameter description of a neural network agent.
It does not contain weights — it describes the *architecture and training configuration*
that you would use to create and train such an agent.

```python
from aumai_metaagent.models import AgentBlueprint, ActivationFn, OptimizerType

# Create a blueprint manually
blueprint = AgentBlueprint(
    blueprint_id="manual_001",
    hidden_layers=[128, 64],       # Two hidden layers
    activation=ActivationFn.RELU,
    optimizer=OptimizerType.ADAM,
    learning_rate=1e-3,
    dropout_rate=0.1,
    memory_size=1000,              # Replay buffer size
    exploration_rate=0.1,          # Epsilon for epsilon-greedy
)

print(blueprint.hidden_layers)      # [128, 64]
print(blueprint.parameter_count())  # Estimated inter-layer parameter count
print(blueprint.blueprint_id)       # "manual_001"
```

### Step 2 — Generate random blueprints

`BlueprintGenerator` creates random, mutated, and crossed-over blueprints.

```python
import random
from aumai_metaagent.core import BlueprintGenerator

rng = random.Random(42)
generator = BlueprintGenerator(rng=rng)

blueprint = generator.random(generation=0)
print(blueprint.hidden_layers)   # e.g. [256, 64, 128]
print(blueprint.activation)      # e.g. ActivationFn.TANH
print(blueprint.learning_rate)   # e.g. 0.000312
```

### Step 3 — Write a fitness function

The fitness function is the core of your system. It takes a blueprint, trains an agent
with those hyperparameters, evaluates it, and returns a scalar fitness score (higher
is better, typically in `[0, 1]`).

For this tutorial, we use a synthetic proxy. In real use, replace this with actual
training and evaluation.

```python
from aumai_metaagent.models import AgentBlueprint

def my_fitness(blueprint: AgentBlueprint) -> float:
    """
    Synthetic fitness: rewards moderate-depth networks with learning rate
    near 1e-3. Replace with real training + evaluation in production.
    """
    # Prefer 2-3 hidden layers
    depth_score = 1.0 / (1.0 + abs(len(blueprint.hidden_layers) - 2))

    # Prefer learning rate near 1e-3
    lr_distance = abs(blueprint.learning_rate - 1e-3) / 1e-3
    lr_score = max(0.0, 1.0 - lr_distance)

    # Prefer low dropout
    dropout_penalty = blueprint.dropout_rate * 0.3

    return max(0.0, min(1.0, depth_score * 0.5 + lr_score * 0.3 - dropout_penalty))
```

### Step 4 — Configure and run evolution

```python
from aumai_metaagent.core import EvolutionEngine
from aumai_metaagent.models import EvolutionConfig

config = EvolutionConfig(
    population_size=20,    # 20 blueprints per generation
    generations=15,        # Evolve for 15 generations
    tournament_size=4,     # 4-way tournament selection
    mutation_rate=0.2,     # 20% per-parameter mutation probability
    crossover_rate=0.7,    # 70% chance of crossover vs. clone
    elite_fraction=0.1,    # Top 10% carry over unchanged
    seed=42,
)

engine = EvolutionEngine(config, fitness_fn=my_fitness)
best = engine.evolve()

print(f"Best blueprint ID : {best.blueprint_id}")
print(f"Fitness           : {best.fitness:.4f}")
print(f"Found in gen      : {best.generation}")
```

### Step 5 — Inspect the evolution history

```python
history = engine.get_history()
print(f"Total evaluations: {len(history)}")  # population_size * generations = 300

# Best fitness per generation
from collections import defaultdict
by_gen: dict[int, list[float]] = defaultdict(list)
for metric in history:
    by_gen[metric.generation].append(metric.fitness)

for gen, fitnesses in sorted(by_gen.items()):
    best_fit = max(fitnesses)
    mean_fit = sum(fitnesses) / len(fitnesses)
    print(f"Gen {gen:3d}: best={best_fit:.4f}  mean={mean_fit:.4f}")
```

### Step 6 — Try the CLI

```bash
# Run evolution with defaults (uses synthetic fitness function)
aumai-metaagent evolve --generations 20 --population 20 --seed 42

# Generate 5 random blueprints as JSON
aumai-metaagent generate --count 5 --seed 0

# Save everything for analysis
aumai-metaagent evolve \
    --generations 50 \
    --population 30 \
    --seed 42 \
    --output best_blueprint.json \
    --history-output evolution_history.json
```

---

## Common Patterns and Recipes

### Pattern 1: Using the real fitness function from a training run

This is the intended production pattern. The fitness function calls your actual training
code and returns a measured quality metric.

```python
from aumai_metaagent.models import AgentBlueprint

def real_fitness(blueprint: AgentBlueprint) -> float:
    """
    Train an agent with the blueprint's hyperparameters and evaluate it.
    Returns success_rate on a held-out evaluation set.

    This is pseudocode — replace with your actual training framework.
    """
    # Build your model using blueprint parameters
    # model = build_model(
    #     layers=blueprint.hidden_layers,
    #     activation=blueprint.activation.value,
    #     dropout=blueprint.dropout_rate,
    # )
    # Train
    # trainer = Trainer(
    #     optimizer=blueprint.optimizer.value,
    #     lr=blueprint.learning_rate,
    #     replay_buffer_size=blueprint.memory_size,
    # )
    # metrics = trainer.train(model, episodes=10)
    # return metrics["success_rate"]

    # Placeholder until your code is integrated:
    return 0.5
```

### Pattern 2: Restarting evolution from a known-good blueprint

If you have a previously discovered good blueprint, you can seed the initial population
with it to warm-start the search.

```python
import random
from aumai_metaagent.core import BlueprintGenerator, EvolutionEngine
from aumai_metaagent.models import AgentBlueprint, EvolutionConfig

# Load a previously evolved blueprint
import json
from pathlib import Path

saved = AgentBlueprint.model_validate_json(Path("best_blueprint.json").read_text())

# Manually seed a population around the saved blueprint
rng = random.Random(99)
generator = BlueprintGenerator(rng=rng)

seed_population = [saved] + [generator.mutate(saved, mutation_rate=0.3) for _ in range(9)]
print(f"Seeded population with {len(seed_population)} blueprints")

# Unfortunately EvolutionEngine does not yet accept a warm-start population;
# this feature is planned. For now, use the generator directly for manual loops.
```

### Pattern 3: Comparing mutation rates

```python
from aumai_metaagent.core import EvolutionEngine
from aumai_metaagent.models import EvolutionConfig

results = {}
for mutation_rate in [0.05, 0.1, 0.2, 0.4]:
    config = EvolutionConfig(
        population_size=20,
        generations=10,
        mutation_rate=mutation_rate,
        seed=0,
    )
    engine = EvolutionEngine(config)
    best = engine.evolve()
    results[mutation_rate] = best.fitness

for rate, fitness in sorted(results.items()):
    print(f"mutation_rate={rate:.2f}: best_fitness={fitness:.4f}")
```

### Pattern 4: Exporting blueprints for downstream use

```python
import json
from pathlib import Path
from aumai_metaagent.models import AgentBlueprint

# Serialise a blueprint to JSON
blueprint_dict = blueprint.model_dump()
Path("blueprint.json").write_text(json.dumps(blueprint_dict, indent=2))

# Reload
loaded = AgentBlueprint.model_validate_json(Path("blueprint.json").read_text())
assert loaded.blueprint_id == blueprint.blueprint_id
assert loaded.hidden_layers == blueprint.hidden_layers

# Use blueprint params to construct a model in your framework
print(f"Layers: {loaded.hidden_layers}")
print(f"LR:     {loaded.learning_rate}")
print(f"Optim:  {loaded.optimizer.value}")   # "adam", "sgd", etc.
print(f"Act:    {loaded.activation.value}")  # "relu", "tanh", etc.
```

### Pattern 5: Analysing what the evolution learned

```python
from collections import Counter
from aumai_metaagent.core import EvolutionEngine
from aumai_metaagent.models import EvolutionConfig

config = EvolutionConfig(population_size=30, generations=20, seed=7)
engine = EvolutionEngine(config)
engine.evolve()

history = engine.get_history()

# What activation functions dominated the best individuals?
top_metrics = sorted(history, key=lambda m: m.fitness, reverse=True)[:20]
top_ids = {m.blueprint_id for m in top_metrics}

# Note: get_history() returns PerformanceMetrics, not blueprints.
# To analyse blueprint traits of top performers, you would need to persist the
# blueprint alongside its metric — a planned feature in v0.2.0.

# For now, analyse the aggregate fitness distribution
all_fitnesses = [m.fitness for m in history]
print(f"Min: {min(all_fitnesses):.4f}")
print(f"Max: {max(all_fitnesses):.4f}")
print(f"Mean: {sum(all_fitnesses)/len(all_fitnesses):.4f}")
```

---

## Troubleshooting FAQ

**Q: Evolution always converges to the same blueprint regardless of seed.**

A: The default fitness function has a single global optimum (depth=2, lr=1e-3, tanh,
Adam). If you use it, convergence is expected. Inject a custom fitness function that
is multimodal to observe genuine evolutionary dynamics.

**Q: Fitness values are all zero after the first generation.**

A: Your fitness function likely raises an exception that is being silently swallowed,
or always returns `0.0`. Add print statements inside your fitness function to debug.
`EvolutionEngine` does not catch exceptions from `fitness_fn` — they will propagate.

**Q: `parameter_count()` returns 0 for a single-layer blueprint.**

A: `parameter_count()` counts inter-layer connections only. A single hidden layer has
no inter-layer connections, so returns 0. This is intentional — the actual input and
output layer sizes are not known at the blueprint level (they depend on your task).

**Q: `EvolutionEngine.evolve()` is slow.**

A: The bottleneck is your `fitness_fn`. The evolution loop itself is pure Python and
very fast. If fitness evaluation involves neural network training, consider reducing
`population_size` and `generations`, or parallelising `fitness_fn` calls with
`concurrent.futures`.

**Q: I want to save the best blueprint, not just the best metric.**

A: The current `evolve()` returns `PerformanceMetric` (which contains `blueprint_id`
but not the full blueprint). The blueprint itself is not persisted by the engine.
Workaround: wrap `fitness_fn` to save each evaluated blueprint by ID:

```python
blueprint_registry: dict[str, AgentBlueprint] = {}

def tracked_fitness(blueprint: AgentBlueprint) -> float:
    blueprint_registry[blueprint.blueprint_id] = blueprint
    return my_fitness(blueprint)

engine = EvolutionEngine(config, fitness_fn=tracked_fitness)
best_metric = engine.evolve()
best_blueprint = blueprint_registry[best_metric.blueprint_id]
```

**Q: How do I interpret `PerformanceMetric.episode_reward`?**

A: In the default fitness function, `episode_reward = fitness * 100` — a simple scaling
of the fitness score. When you inject your own `fitness_fn`, you control what `fitness`
means; `episode_reward` and `success_rate` in the metric are set by the engine from
the `fitness` value via the same formula unless you override them (planned feature).
For now, treat `fitness` as the primary signal and the other fields as informational
only when using the default engine.
