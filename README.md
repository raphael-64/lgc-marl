# LGC-MARL with Progressive Depth Evolution on RWARE

Implementation of LGC-MARL (LLM-based Graph Collaboration MARL) with Progressive Depth Evolution for multi-robot warehouse coordination.

## Overview

This project combines:
- **LGC-MARL**: LLM-generated task decomposition graphs for multi-agent coordination
- **Progressive Depth Evolution**: Efficient graph search through staged elimination and evolution
- **RWARE**: Multi-Robot Warehouse environment for evaluation

### Key Features

- **LLM Graph Generation**: GPT-4o-mini (or local models) generates task decomposition graphs
- **Graph-Conditioned Policies**: Neural policies that adapt to different graph structures
- **Progressive Depth**: Start with many candidates + few episodes, evolve to fewer candidates + more episodes
- **Policy Transfer**: Transfer learned weights between graph structures
- **Weave + W&B Integration**: Full observability and experiment tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROGRESSIVE DEPTH EVOLUTION                    │
│                                                                  │
│  Stage 1: 8 candidates × 50 eps   ──▶  Evolve                   │
│  Stage 2: 4 candidates × 150 eps  ──▶  Evolve                   │
│  Stage 3: 2 candidates × 400 eps  ──▶  Evolve                   │
│  Stage 4: 1 candidate × 400 eps   ──▶  Best Graph               │
│                                                                  │
│  Total: 2200 episodes (vs 8000 naive)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd lgc-marl-rware

# Create environment
conda create -n lgc-marl python=3.11
conda activate lgc-marl

# Install package
pip install -e ".[dev]"

# Install RWARE
pip install rware
```

### API Keys

Set your OpenAI API key for graph generation:
```bash
export OPENAI_API_KEY=your_key_here
```

## Quick Start

```bash
# Run training with default config
python scripts/train.py

# With custom environment
python scripts/train.py env.name=rware-tiny-2ag-v2

# With different difficulty
python scripts/train.py task.difficulty=hard

# With custom evolution stages
python scripts/train.py \
    evolution.stages='[{candidates: 4, episodes: 100}, {candidates: 2, episodes: 300}, {candidates: 1, episodes: 600}]'
```

## Configuration

Main configuration in `configs/default.yaml`:

```yaml
# Environment
env:
  name: "rware-small-4ag-v2"
  max_steps: 500

# Task
task:
  difficulty: "medium"  # easy, medium, hard

# LLM for graph generation
llm:
  model: "gpt-4o-mini"
  temperature: 0.7

# Evolution stages
evolution:
  stages:
    - candidates: 8
      episodes: 50
    - candidates: 4
      episodes: 150
    - candidates: 2
      episodes: 400
    - candidates: 1
      episodes: 400
```

## Project Structure

```
lgc-marl-rware/
├── src/
│   ├── graph_generation/     # LLM graph planning
│   │   ├── graph_types.py    # TaskGraph, Subtask data structures
│   │   ├── llm_planner.py    # LLM-based graph generator
│   │   └── prompts.py        # Prompt templates
│   │
│   ├── lgc_marl/            # Core MARL components
│   │   ├── graph_policy.py   # Graph-conditioned policy network
│   │   └── marl_trainer.py   # Multi-agent PPO trainer
│   │
│   ├── evolution/           # Progressive depth evolution
│   │   ├── progressive_depth.py  # Main evolution loop
│   │   ├── selection.py          # Selection strategies
│   │   └── policy_transfer.py    # Policy transfer utilities
│   │
│   ├── environments/        # Environment wrappers
│   │   └── rware_wrapper.py # RWARE wrapper with graph support
│   │
│   └── tracking/           # Logging utilities
│       ├── weave_ops.py    # Weave integration
│       └── metrics.py      # W&B metrics
│
├── scripts/
│   ├── train.py            # Main training script
│   └── evaluate.py         # Evaluation script
│
├── configs/
│   └── default.yaml        # Default configuration
│
└── tests/                  # Unit tests
```

## How It Works

### 1. Graph Generation

The LLM generates task decomposition graphs with different strategies:
- Sequential, Parallel, Pipelined, Zone-based, etc.

Example graph:
```
- fetch_shelf1: Agent_0 fetch shelf_1
- fetch_shelf2: Agent_1 fetch shelf_2
- deliver_shelf1: Agent_0 deliver workstation (after: fetch_shelf1)
- deliver_shelf2: Agent_1 deliver workstation (after: fetch_shelf2, deliver_shelf1)
```

### 2. Graph-Conditioned Policy

The policy network takes:
- Per-agent observations
- Graph embedding (from GNN encoder)

And outputs actions for each agent.

### 3. Progressive Depth Evolution

1. **Stage 1**: Generate 8 diverse graphs, train each for 50 episodes
2. **Evaluate**: Measure success rate for each
3. **Evolve**: Keep top 2 (elites), generate 2 new (crossover/mutation)
4. **Stage 2**: Train 4 candidates for 150 episodes
5. **Repeat** until convergence

### 4. Policy Transfer

When evolving graphs, policies are transferred:
- Clone parent policy weights
- Add small noise for exploration
- Continue training on new graph

## Evaluation

```bash
# Evaluate trained policy
python scripts/evaluate.py --checkpoint-dir ./checkpoints --n-episodes 100

# With rendering
python scripts/evaluate.py --checkpoint-dir ./checkpoints --render
```

## Compute Requirements

| Configuration | Episodes | Est. Time (4× A100) |
|--------------|----------|---------------------|
| Default (8→4→2→1) | 2200 | ~55 min |
| Fast (4→2→1) | 1100 | ~30 min |
| Thorough (16→8→4→2→1) | 4400 | ~2 hours |

## Extending

### Custom Selection Strategy

```python
from src.evolution.selection import SelectionStrategy

class MySelection(SelectionStrategy):
    def select(self, candidates, policies, performances, n_select):
        # Your selection logic
        return selected_indices
```

### Custom Environment

```python
from src.environments.rware_wrapper import RWAREGraphWrapper

class MyEnvWrapper(RWAREGraphWrapper):
    def _check_subtask_completion(self, info):
        # Custom subtask completion logic
        pass
```

## References

- [LGC-MARL Paper](https://arxiv.org/abs/2503.10049)
- [RWARE Environment](https://github.com/uoe-agents/robotic-warehouse)
- [Weave by W&B](https://docs.wandb.ai/weave/)

## License

MIT
