# Learning Environmental Policy in a Multi-Agent Reinforcement Learning Framework

This project models environmental regulation as a multi-agent reinforcement learning problem, where a government agent learns to set taxes or emission quotas to maximise social welfare, while firm agents learn to adapt their production decisions accordingly.

## Structure

- `env.py` — Base ecosystem environment
- `env_penalty.py` — Environment with catastrophe penalty and pollution cap
- `agents.py` — Q-Learning government and firm agents
- `agents_penalty.py` — Agents with termination-aware learning 
- `main.ipynb` — Main notebook: training, evaluation, and sensitivity analysis

## Run

The notebook takes approximately **2h30** to execute in full.


## Methods

- **Section I** — Q-Learning for all agents
- **Section II** — Q-Learning vs Expected SARSA under termination constraints 
- **Section III** — Sensitivity analysis over λ, ρ, p, and α

## Results

Full analysis and interpretation of results are provided in the  report.
