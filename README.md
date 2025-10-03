# ♟️ Game Theory Decision Engine – Strategic Decision Simulator

A Python-based simulation project that demonstrates how **game theory**, **probability**, and **decision logic** can be applied to trading and strategic decision-making.  
It includes payoff matrix modeling, **mixed-strategy Nash equilibrium** calculation, **repeated game simulations** with learning (Fictitious Play, ε-greedy), and a simple **sequential decision tree** for buy/hold/sell choices.  
This project connects theoretical mathematics with real-world decision-making in financial markets — making it ideal for portfolio use in trading, AI, or algorithmic reasoning roles.

---

## 📊 Project Overview

This project was created to explore how code can represent and solve decision-making problems similar to those encountered in trading or strategic environments.  
It is designed as a **beginner → intermediate** portfolio project with a strong focus on:

- 🎲 **Game theory modeling** – 2×2 normal-form games with payoffs
- ♟️ **Nash equilibrium** – compute optimal mixed strategies
- 🔁 **Repeated games** – simulate adaptive strategies using learning algorithms
- 🌳 **Decision trees** – explore sequential buy/hold/sell decisions
- 📈 **Monte Carlo simulations** – estimate long-term optimal policies

---

## ✅ Features

- **Payoff matrix simulation**: Define and analyze 2×2 strategic games
- **Mixed-strategy Nash equilibrium** calculation
- **Fictitious Play** simulation: players best respond to opponent’s historical actions
- **ε-greedy learning**: balance exploration and exploitation in repeated plays
- **Sequential decision tree** for trading logic (buy / hold / sell)
- **Monte Carlo rollouts** to estimate expected long-term rewards
- **CSV export** of results (optional)
- **Extensive console reporting** with payoff summaries and equilibrium analysis

---

## 📂 Example Use Cases

- Simulate basic strategic interactions between two agents (e.g., market maker vs trader)
- Analyze and visualize mixed strategies in repeated competitive games
- Test how exploration impacts performance with ε-greedy learning
- Build a decision tree for trading scenarios and compute expected rewards
- Use Monte Carlo rollouts to estimate long-run outcomes of strategic policies

---

## 🧠 Core Concepts Demonstrated

| Concept | Description |
|--------|-------------|
| **Nash Equilibrium** | A state where no player can improve their payoff by changing their strategy unilaterally. |
| **Fictitious Play** | A learning method where players respond optimally to the empirical frequency of opponent actions. |
| **ε-greedy Strategy** | A reinforcement learning technique that chooses the best-known action most of the time, but explores alternatives with probability ε. |
| **Decision Trees** | A structured model of sequential decisions under uncertainty. |
| **Monte Carlo Simulation** | A statistical method to approximate long-term performance by running many random trials. |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Core Libraries:** `numpy`, `pandas`, `random`
- *(Optional)* for visualization: `matplotlib`, `seaborn`

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/game-theory-decision-engine.git
cd game-theory-decision-engine
pip install -r requirements.txt
