# aumai-metaagent API Reference

Evolutionary meta-learning engine that learns to build better agents.
Populations of `AgentBlueprint` objects evolve across generations using
tournament selection, crossover, and mutation. This reference documents every
public class, method, and Pydantic model in the package.

---

## Table of Contents

1. [Models](#models)
   - [ActivationFn](#activationfn)
   - [OptimizerType](#optimizertype)
   - [AgentBlueprint](#agentblueprint)
   - [PerformanceMetric](#performancemetric)
   - [EvolutionConfig](#evolutionconfig)
2. [Core Classes](#core-classes)
   - [BlueprintGenerator](#blueprintgenerator)
   - [EvolutionEngine](#evolutionengine)
3. [CLI Commands](#cli-commands)
4. [Package Exports](#package-exports)

---

## Models

All models are Pydantic v2 `BaseModel` subclasses. Import from
`aumai_metaagent.models`.

---

### ActivationFn

String enumeration of supported activation functions for neural network
hidden layers.

```python
from aumai_metaagent.models import ActivationFn

fn = ActivationFn.RELU
print(fn.value)  # "relu"
```

#### Members

| Member | Value | Description |
|---|---|---|
| `RELU` | `"relu"` | Rectified Linear Unit. |
| `TANH` | `"tanh"` | Hyperbolic tangent. |
| `SIGMOID` | `"sigmoid"` | Logistic sigmoid. |
| `LINEAR` | `"linear"` | No non-linearity. |
| `GELU` | `"gelu"` | Gaussian Error Linear Unit. |

---

### OptimizerType

String enumeration of supported optimizers.

```python
from aumai_metaagent.models import OptimizerType

opt = OptimizerType.ADAM
print(opt.value)  # "adam"
```

#### Members

| Member | Value | Description |
|---|---|---|
| `ADAM` | `"adam"` | Adaptive Moment Estimation. |
| `SGD` | `"sgd"` | Stochastic Gradient Descent. |
| `RMSPROP` | `"rmsprop"` | RMSProp. |
| `ADAMW` | `"adamw"` | Adam with decoupled weight decay. |

---

### AgentBlueprint

A parameterised description of an agent architecture. Each instance
represents one candidate in the evolutionary population.

```python
from aumai_metaagent.models import AgentBlueprint, ActivationFn, OptimizerType

blueprint = AgentBlueprint(
    blueprint_id="bp_a1b2c3d4",
    hidden_layers=[128, 64],
    activation=ActivationFn.RELU,
    optimizer=OptimizerType.ADAM,
    learning_rate=1e-3,
    dropout_rate=0.1,
    memory_size=1000,
    exploration_rate=0.1,
    generation=0,
)
```

#### Fields

| Field | Type | Required | Constraint | Default | Description |
|---|---|---|---|---|---|
| `blueprint_id` | `str` | Yes | — | — | Unique identifier (format: `"bp_<8-char hex>"`). |
| `hidden_layers` | `list[int]` | No | — | `[64, 64]` | List of hidden layer sizes (number of neurons per layer). |
| `activation` | `ActivationFn` | No | — | `RELU` | Activation function for hidden layers. |
| `optimizer` | `OptimizerType` | No | — | `ADAM` | Optimizer type. |
| `learning_rate` | `float` | No | `> 0.0` | `0.001` | Learning rate for the optimizer. |
| `dropout_rate` | `float` | No | `0.0 <= x <= 0.9` | `0.0` | Dropout regularisation rate. |
| `memory_size` | `int` | No | `> 0` | `1000` | Replay buffer or context window size. |
| `exploration_rate` | `float` | No | `0.0 <= x <= 1.0` | `0.1` | Epsilon for epsilon-greedy exploration. |
| `generation` | `int` | No | `>= 0` | `0` | Evolution generation in which this blueprint was created. |
| `parent_ids` | `list[str]` | No | — | `[]` | Blueprint IDs of the parent(s) used to produce this individual. |
| `extra_params` | `dict[str, Any]` | No | — | `{}` | Extensible additional hyperparameters. |

#### Methods

##### `AgentBlueprint.parameter_count`

```python
def parameter_count(self) -> int
```

Estimate the total trainable parameter count implied by `hidden_layers`.
Counts weight matrices plus bias vectors for each consecutive pair of layers.

**Returns**

`int` — Estimated parameter count. Returns `0` if `hidden_layers` has fewer
than two entries.

**Example**

```python
bp = AgentBlueprint(blueprint_id="bp_test", hidden_layers=[128, 64, 32])
# (128*64 + 64) + (64*32 + 32) = 8256 + 2080 = 10336
print(bp.parameter_count())  # 10336
```

---

### PerformanceMetric

A measured performance outcome for an `AgentBlueprint` after evaluation.
Produced by `EvolutionEngine.evolve` and stored in the evolution history.

```python
from aumai_metaagent.models import PerformanceMetric

metric = PerformanceMetric(
    blueprint_id="bp_a1b2c3d4",
    fitness=0.82,
    episode_reward=82.0,
    success_rate=0.75,
    generation=3,
)
```

#### Fields

| Field | Type | Required | Constraint | Default | Description |
|---|---|---|---|---|---|
| `blueprint_id` | `str` | Yes | — | — | Blueprint this metric belongs to. |
| `fitness` | `float` | No | — | `0.0` | Primary fitness score. Higher is better. |
| `episode_reward` | `float` | No | — | `0.0` | Mean reward over evaluation episodes. |
| `success_rate` | `float` | No | `0.0 <= x <= 1.0` | `0.0` | Fraction of tasks completed successfully. |
| `evaluation_steps` | `int` | No | `>= 0` | `0` | Number of environment steps used for evaluation. |
| `generation` | `int` | No | `>= 0` | `0` | Generation at which this metric was measured. |
| `notes` | `str \| None` | No | — | `None` | Optional contextual notes or diagnostic information. |

---

### EvolutionConfig

Hyperparameters for the evolutionary meta-learning loop. Passed to
`EvolutionEngine` at construction time.

```python
from aumai_metaagent.models import EvolutionConfig

config = EvolutionConfig(
    population_size=30,
    generations=50,
    tournament_size=5,
    mutation_rate=0.2,
    crossover_rate=0.7,
    elite_fraction=0.1,
    seed=42,
    evaluation_episodes=10,
)
```

#### Fields

| Field | Type | Required | Constraint | Default | Description |
|---|---|---|---|---|---|
| `population_size` | `int` | No | `> 0` | `20` | Number of blueprints per generation. |
| `generations` | `int` | No | `> 0` | `10` | Total number of evolution generations to run. |
| `tournament_size` | `int` | No | `> 0` | `4` | Number of individuals in each tournament selection. Capped at `population_size`. |
| `mutation_rate` | `float` | No | `0.0 <= x <= 1.0` | `0.2` | Per-parameter probability of mutation. |
| `crossover_rate` | `float` | No | `0.0 <= x <= 1.0` | `0.7` | Probability of crossover vs. cloning when producing offspring. |
| `elite_fraction` | `float` | No | `0.0 <= x <= 1.0` | `0.1` | Fraction of top individuals carried unchanged to the next generation. At least 1 elite is always kept. |
| `seed` | `int \| None` | No | — | `None` | Optional random seed for reproducibility. |
| `evaluation_episodes` | `int` | No | `> 0` | `10` | Episodes per fitness evaluation (available to custom fitness functions). |

---

## Core Classes

Import from `aumai_metaagent.core`.

---

### BlueprintGenerator

Generates, mutates, and combines `AgentBlueprint` objects. All randomness
flows through a single `random.Random` instance, making results reproducible
when a seeded RNG is provided.

```python
import random
from aumai_metaagent.core import BlueprintGenerator

rng = random.Random(42)
generator = BlueprintGenerator(rng=rng)
```

#### Constructor

```python
BlueprintGenerator(rng: random.Random | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rng` | `random.Random \| None` | `None` | Seeded RNG instance. If `None`, a new `random.Random()` is created with system entropy. |

#### Architecture search space

| Hyperparameter | Choices |
|---|---|
| Number of hidden layers | 1 to 4 |
| Hidden layer size | 16, 32, 64, 128, 256, 512 |
| Activation | All `ActivationFn` members |
| Optimizer | All `OptimizerType` members |
| Learning rate | Log-uniform in [1e-5, 1e-1] |
| Dropout rate | Uniform in [0.0, 0.5] |
| Memory size | 256, 512, 1000, 2000, 5000 |
| Exploration rate | Uniform in [0.01, 0.5] |

#### `BlueprintGenerator.random`

```python
def random(self, generation: int = 0) -> AgentBlueprint
```

Generate a fully random `AgentBlueprint` by sampling uniformly from the
search space.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `generation` | `int` | `0` | Generation number to stamp on the blueprint. |

**Returns**

`AgentBlueprint` — A randomly parameterised blueprint with a unique
`blueprint_id` in the format `"bp_<8-char hex>"`.

**Example**

```python
blueprint = generator.random(generation=0)
print(blueprint.blueprint_id)    # e.g. "bp_3f7a1c2d"
print(blueprint.hidden_layers)   # e.g. [256, 128]
print(blueprint.activation)      # e.g. ActivationFn.RELU
```

#### `BlueprintGenerator.mutate`

```python
def mutate(self, blueprint: AgentBlueprint, mutation_rate: float = 0.2) -> AgentBlueprint
```

Return a mutated copy of `blueprint`. Each hyperparameter is independently
mutated with probability `mutation_rate`. The original blueprint is not
modified.

Architecture mutations are one of:

- **resize**: Replace a randomly chosen layer with a new random size.
- **add**: Append a new random layer (only if `len(hidden_layers) < 4`).
- **remove**: Drop a random layer (only if `len(hidden_layers) > 1`).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `blueprint` | `AgentBlueprint` | Required | The blueprint to mutate. |
| `mutation_rate` | `float` | `0.2` | Per-parameter mutation probability. |

**Returns**

`AgentBlueprint` — A new mutated blueprint with a fresh `blueprint_id`.
`parent_ids` is set to `[blueprint.blueprint_id]`.

**Example**

```python
original = generator.random(generation=0)
mutated = generator.mutate(original, mutation_rate=0.3)
assert mutated.blueprint_id != original.blueprint_id
assert original.blueprint_id in mutated.parent_ids
```

#### `BlueprintGenerator.crossover`

```python
def crossover(self, parent_a: AgentBlueprint, parent_b: AgentBlueprint) -> AgentBlueprint
```

Create a child blueprint by crossing over two parents using uniform crossover.
Each scalar hyperparameter is independently drawn from either parent with
50% probability. Hidden layers are combined by pairing up positions from both
parents and randomly extending from the longer parent.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `parent_a` | `AgentBlueprint` | First parent blueprint. |
| `parent_b` | `AgentBlueprint` | Second parent blueprint. |

**Returns**

`AgentBlueprint` — A new child blueprint. `generation` is set to
`max(parent_a.generation, parent_b.generation)`. `parent_ids` is set to
`[parent_a.blueprint_id, parent_b.blueprint_id]`.

**Example**

```python
parent_a = generator.random(generation=2)
parent_b = generator.random(generation=2)
child = generator.crossover(parent_a, parent_b)
assert set(child.parent_ids) == {parent_a.blueprint_id, parent_b.blueprint_id}
```

---

### EvolutionEngine

Runs the evolutionary meta-learning loop using tournament selection, elitism,
configurable crossover, and mutation. The fitness function is injected at
construction time to allow real-environment evaluation.

```python
from aumai_metaagent.core import EvolutionEngine
from aumai_metaagent.models import EvolutionConfig

config = EvolutionConfig(population_size=20, generations=10, seed=42)
engine = EvolutionEngine(config)
best = engine.evolve()
```

#### Constructor

```python
EvolutionEngine(
    config: EvolutionConfig,
    fitness_fn: Callable[[AgentBlueprint], float] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `EvolutionConfig` | Required | Evolution hyperparameters. |
| `fitness_fn` | `Callable[[AgentBlueprint], float] \| None` | `None` | Custom fitness function. If `None`, a synthetic proxy is used. |

**Default fitness function** (used when `fitness_fn` is `None`):

Rewards moderate-depth networks (2 hidden layers), learning rates near 1e-3,
`TANH` activation (+0.1 bonus), and `ADAM` optimizer (+0.05 bonus). Penalises
high dropout. Returns a value in approximately [0, 1]. Suitable only for
demonstration; replace with real environment evaluation in production.

#### `EvolutionEngine.evolve`

```python
def evolve(self) -> PerformanceMetric
```

Run the full evolution loop for `config.generations` generations and return
the best `PerformanceMetric` found.

**Algorithm per generation:**

1. Evaluate all blueprints with `fitness_fn`.
2. Sort population by fitness descending.
3. Carry the top `elite_fraction` individuals unchanged.
4. Fill the rest by tournament selection followed by crossover (probability
   `crossover_rate`) and mutation (probability `mutation_rate`).

**Returns**

`PerformanceMetric` — Best metric observed across all generations. If no
updates were ever scored (degenerate config), returns a zero-fitness sentinel.

**Example**

```python
config = EvolutionConfig(population_size=30, generations=25, seed=7)
engine = EvolutionEngine(config)
best = engine.evolve()
print(f"Best blueprint: {best.blueprint_id}")
print(f"Fitness: {best.fitness:.4f}")
print(f"Found in generation: {best.generation}")
```

#### `EvolutionEngine.get_history`

```python
def get_history(self) -> list[PerformanceMetric]
```

Return all `PerformanceMetric` records produced during the evolution run,
one per (blueprint, generation) evaluation. Useful for plotting fitness curves
or post-run analysis.

**Returns**

`list[PerformanceMetric]` — Flat list of all evaluated metrics in evaluation
order. Returns a copy; mutations do not affect the engine's internal state.

**Example**

```python
history = engine.get_history()
# Total evaluations = population_size * generations
print(f"Total evaluations: {len(history)}")

# Find average fitness per generation
from collections import defaultdict
by_gen: dict[int, list[float]] = defaultdict(list)
for metric in history:
    by_gen[metric.generation].append(metric.fitness)
for gen, fitnesses in sorted(by_gen.items()):
    print(f"Gen {gen:3d}  avg_fitness={sum(fitnesses)/len(fitnesses):.4f}")
```

---

## CLI Commands

The `aumai-metaagent` entry point groups two sub-commands. All accept
`--help` for full option descriptions.

```
aumai-metaagent [--version] [--help] COMMAND [ARGS]...
```

---

### `evolve`

Run the full evolutionary meta-learning loop and save the best blueprint to
a JSON file.

```
aumai-metaagent evolve [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--generations INT` | `int` | `10` | Number of evolution generations. |
| `--population INT` | `int` | `20` | Population size per generation. |
| `--tournament-size INT` | `int` | `4` | Tournament selection size. |
| `--mutation-rate FLOAT` | `float` | `0.2` | Per-parameter mutation probability. |
| `--crossover-rate FLOAT` | `float` | `0.7` | Crossover vs. cloning probability. |
| `--elite-fraction FLOAT` | `float` | `0.1` | Fraction of elite survivors. |
| `--seed INT` | `int` | `None` | Random seed for reproducibility. |
| `--output PATH` | `Path` | `evolution_result.json` | Output path for the best blueprint JSON. |
| `--history-output PATH` | `Path` | `None` | Optional path to save the full evaluation history JSON. |

**Example**

```bash
aumai-metaagent evolve --generations 100 --population 30 --seed 42 \
    --output best.json --history-output history.json
```

**Output JSON structure** (`--output`): serialised `PerformanceMetric` model.

**History JSON structure** (`--history-output`): array of serialised
`PerformanceMetric` objects, one per evaluation.

---

### `generate`

Generate one or more random `AgentBlueprint` objects and print them as JSON
or write them to a file.

```
aumai-metaagent generate [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--count INT` | `int` | `1` | Number of blueprints to generate. |
| `--seed INT` | `int` | `None` | Random seed for reproducibility. |
| `--output PATH` | `Path` | `None` | Output JSON path. Defaults to stdout if not provided. |

**Example**

```bash
# Print 5 blueprints to stdout
aumai-metaagent generate --count 5 --seed 0

# Write to file
aumai-metaagent generate --count 10 --seed 99 --output blueprints.json
```

**Output JSON structure**: array of serialised `AgentBlueprint` objects.

---

## Package Exports

```python
import aumai_metaagent
print(aumai_metaagent.__version__)  # "0.1.0"
```

Public symbols (import from `aumai_metaagent.core` and `aumai_metaagent.models`):

```python
from aumai_metaagent.core import BlueprintGenerator, EvolutionEngine
from aumai_metaagent.models import (
    ActivationFn,
    AgentBlueprint,
    EvolutionConfig,
    OptimizerType,
    PerformanceMetric,
)
```
