# repo_name
more to come...

## CartPole Policy Gradient (REINFORCE)

`cartpole_pg.py` implements a policy-gradient agent with PyTorch that learns
Gymnasium's `CartPole-v1` from scratch. A **run** and an **episode** are the same
thing: one attempt at balancing the pole, from reset until it falls or the
500-step time limit is reached.

### Install (uv)

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

This creates a `.venv` and installs everything from the lockfile. Run all the
commands below through `uv run` so they use that environment. If `uv` is not on
your `PATH`, replace `uv` with `python3 -m uv`.

`pygame` is used to display the live `play` window; `imageio` and `pillow` are
needed only for the `gif` command.

### 1. Train from scratch

```bash
uv run python cartpole_pg.py train
# custom example:
uv run python cartpole_pg.py train --episodes 1000 --learning-rate 0.01 --gamma 0.99 --seed 42 --output cartpole_policy.pt
```

Training samples a full episode with the current policy, computes discounted
reward-to-go for each action, normalizes those returns, and updates the network
with the REINFORCE loss. Because training can degrade late, the weights with the
best 100-run average reward are kept. Those weights **and** the network
configuration are saved to `cartpole_policy.pt` by default. Use `--help` on any
command to see all options (hidden layer size, learning rate, discount, seed,
log interval, ...).

### 2. Watch a trained agent play

```bash
uv run python cartpole_pg.py play                       # opens a render window, 5 runs
uv run python cartpole_pg.py play --episodes 10 --seed 7
uv run python cartpole_pg.py play --no-render           # headless: only print rewards
```

Each run prints its total reward and number of steps, followed by the average.
Actions are chosen greedily from the trained policy (no exploration).

### 3. Record a run as a GIF

```bash
uv run python cartpole_pg.py gif
# custom example:
uv run python cartpole_pg.py gif --model cartpole_policy.pt --output cartpole_run.gif --fps 30 --seed 7
```

This records exactly one run as `cartpole_run.gif` by default. The `--seed`
option makes a particular run reproducible.

## Useful references
- huggingface on Policy Gradient with PyTorch: https://huggingface.co/learn/deep-rl-course/unit4/introduction
- Gym documentation: https://gymnasium.farama.org/
- Working with gym in PyTorch: https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/GYM.html
- Lunar Lander the game: https://gymnasium.farama.org/environments/box2d/lunar_lander/