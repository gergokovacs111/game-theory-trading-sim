#!/usr/bin/env python3
# nash_decision_simulator.py
# Beginner-friendly Game Theory Decision Engine (2x2 games + simple decision trees)
# Author: Gergő Kovács
# License: MIT

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import numpy as np
except ImportError:
    np = None  # The code works without NumPy; it's only used for optional helpers.


# =========================
# Data Models
# =========================

@dataclass(frozen=True)
class NormalForm2x2:
    """
    A 2x2 normal-form game with Row player (R) and Column player (C).

    Payoff matrices:
        R: [[r11, r12],
            [r21, r22]]
        C: [[c11, c12],
            [c21, c22]]

    Actions:
        Row:    0 -> 'R0', 1 -> 'R1'
        Column: 0 -> 'C0', 1 -> 'C1'
    """
    R: List[List[float]]
    C: List[List[float]]
    row_labels: Tuple[str, str] = ("R0", "R1")
    col_labels: Tuple[str, str] = ("C0", "C1")

    def best_response_row(self, q: float) -> int:
        """
        Best response of Row to Column mixing with q = P(C chooses col 0).
        Row expected payoffs:
            U(R0) = q*R[0][0] + (1-q)*R[0][1]
            U(R1) = q*R[1][0] + (1-q)*R[1][1]
        """
        u_r0 = q * self.R[0][0] + (1 - q) * self.R[0][1]
        u_r1 = q * self.R[1][0] + (1 - q) * self.R[1][1]
        return 0 if u_r0 >= u_r1 else 1

    def best_response_col(self, p: float) -> int:
        """
        Best response of Column to Row mixing with p = P(R chooses row 0).
        Column expected payoffs:
            U(C0) = p*C[0][0] + (1-p)*C[1][0]
            U(C1) = p*C[0][1] + (1-p)*C[1][1]
        """
        u_c0 = p * self.C[0][0] + (1 - p) * self.C[1][0]
        u_c1 = p * self.C[0][1] + (1 - p) * self.C[1][1]
        return 0 if u_c0 >= u_c1 else 1

    def expected_payoffs(self, p: float, q: float) -> Tuple[float, float]:
        """
        Expected payoffs to (Row, Column) when Row mixes with p over row 0,
        and Column mixes with q over col 0.
        """
        # Probabilities over outcomes:
        probs = {
            (0, 0): p * q,
            (0, 1): p * (1 - q),
            (1, 0): (1 - p) * q,
            (1, 1): (1 - p) * (1 - q),
        }
        u_r = sum(self.R[i][j] * prob for (i, j), prob in probs.items())
        u_c = sum(self.C[i][j] * prob for (i, j), prob in probs.items())
        return u_r, u_c


# =========================
# Nash Equilibrium (2x2 Mixed)
# =========================

def solve_mixed_nash_2x2(game: NormalForm2x2) -> Optional[Tuple[float, float]]:
    """
    Returns (p*, q*), the mixed-strategy Nash equilibrium for a 2x2 game,
    where p* = P(Row plays row 0), q* = P(Col plays col 0).

    Derivation:
      Indifference conditions:
        Row indifferent => q* makes U(R0) == U(R1)
        Column indifferent => p* makes U(C0) == U(C1)

      Solve:
        q* = (R[1][1] - R[0][1]) / ((R[0][0] - R[1][0]) + (R[1][1] - R[0][1]))
        p* = (C[1][1] - C[1][0]) / ((C[0][0] - C[0][1]) + (C[1][1] - C[1][0]))

    Returns None if there is no interior solution in [0,1]^2.
    """
    r = game.R
    c = game.C

    denom_q = (r[0][0] - r[1][0]) + (r[1][1] - r[0][1])
    denom_p = (c[0][0] - c[0][1]) + (c[1][1] - c[1][0])

    if denom_q == 0 or denom_p == 0:
        return None

    q_star = (r[1][1] - r[0][1]) / denom_q
    p_star = (c[1][1] - c[1][0]) / denom_p

    if 0 <= p_star <= 1 and 0 <= q_star <= 1:
        return (p_star, q_star)
    return None


# =========================
# Learning Dynamics
# =========================

def fictitious_play(
    game: NormalForm2x2,
    iterations: int = 5000,
    seed: Optional[int] = 42
) -> Tuple[float, float]:
    """
    Fictitious play for 2x2: each player best-responds to opponent's empirical frequency.
    Returns empirical (p_hat, q_hat) after iterations.
    """
    if seed is not None:
        random.seed(seed)

    # Start with uniform beliefs
    count_row0 = 1
    count_col0 = 1
    t = 2

    for _ in range(iterations):
        q_hat = count_col0 / t
        r_br = game.best_response_row(q_hat)

        p_hat = count_row0 / t
        c_br = game.best_response_col(p_hat)

        # Draw actions as pure best responses:
        if r_br == 0:
            count_row0 += 1
        if c_br == 0:
            count_col0 += 1
        t += 1

    p_final = count_row0 / t
    q_final = count_col0 / t
    return p_final, q_final


def epsilon_greedy_simulation(
    game: NormalForm2x2,
    epsilon: float = 0.1,
    rounds: int = 10000,
    seed: Optional[int] = 7
) -> Dict[str, float]:
    """
    ε-greedy repeated play:
      - With prob (1 - ε), play best response to last observed opponent move.
      - With prob ε, explore (random action).

    Tracks average payoffs and empirical mix.

    Returns:
        {
          "p_row0": freq Row played row 0,
          "q_col0": freq Col played col 0,
          "u_row": avg Row payoff,
          "u_col": avg Col payoff
        }
    """
    if seed is not None:
        random.seed(seed)

    # Initialize with random last moves:
    last_row = random.choice([0, 1])
    last_col = random.choice([0, 1])

    row0_count = 0
    col0_count = 0
    u_row_sum = 0.0
    u_col_sum = 0.0

    for _ in range(rounds):
        # Best responses to *last observed* action (myopic)
        r_br_to_last = (
            0 if game.R[0][last_col] >= game.R[1][last_col] else 1
        )
        c_br_to_last = (
            0 if game.C[last_row][0] >= game.C[last_row][1] else 1
        )

        # ε-exploration:
        row_action = r_br_to_last if random.random() > epsilon else random.choice([0, 1])
        col_action = c_br_to_last if random.random() > epsilon else random.choice([0, 1])

        # Payoffs
        u_r = game.R[row_action][col_action]
        u_c = game.C[row_action][col_action]
        u_row_sum += u_r
        u_col_sum += u_c

        # Update counts and last observed
        if row_action == 0:
            row0_count += 1
        if col_action == 0:
            col0_count += 1
        last_row, last_col = row_action, col_action

    return {
        "p_row0": row0_count / rounds,
        "q_col0": col0_count / rounds,
        "u_row": u_row_sum / rounds,
        "u_col": u_col_sum / rounds,
    }


# =========================
# Simple Decision Tree (Sequential)
# =========================

@dataclass
class DecisionNode:
    """
    A minimal decision node for sequential problems.
    Each action maps to either another DecisionNode or a terminal payoff.
    """
    name: str
    actions: Dict[str, "DecisionNode | Tuple[float, float]"]  # (u_row, u_col) at terminal

    def evaluate(self, policy: Dict[str, str]) -> Tuple[float, float]:
        """
        Follow a deterministic policy dict: {node_name: chosen_action}.
        Returns terminal (u_row, u_col).
        """
        node: DecisionNode = self
        while True:
            action = policy.get(node.name)
            if action is None or action not in node.actions:
                raise ValueError(f"No valid action for node '{node.name}' in policy.")
            next_ = node.actions[action]
            if isinstance(next_, DecisionNode):
                node = next_
            else:
                return next_  # terminal payoff


def build_sample_market_tree() -> DecisionNode:
    """
    A toy sequential decision:
      Node 'MarketState': choose between 'Bullish' or 'Bearish' branch.
      Each branch leads to Row (Trader) decision 'Buy'/'Hold' with different payoffs.
    """
    bullish_leaf = DecisionNode(
        name="BullishDecision",
        actions={
            "Buy": (2.0, 1.0),
            "Hold": (0.5, 0.5),
        },
    )
    bearish_leaf = DecisionNode(
        name="BearishDecision",
        actions={
            "Buy": (-1.0, 1.5),
            "Hold": (0.4, 0.4),
        },
    )
    root = DecisionNode(
        name="MarketState",
        actions={
            "Bullish": bullish_leaf,
            "Bearish": bearish_leaf,
        },
    )
    return root


# =========================
# Demo Scenarios
# =========================

def demo_prisoners_dilemma() -> NormalForm2x2:
    """
    Classic Prisoner's Dilemma (Row, Column):
        Actions: Cooperate (0), Defect (1)
        Payoffs:
         - Mutual C: (3, 3)
         - Mutual D: (1, 1)
         - Tempt (Row D, Col C): (5, 0)
         - Sucker (Row C, Col D): (0, 5)
    """
    R = [[3, 0],
         [5, 1]]
    C = [[3, 5],
         [0, 1]]
    return NormalForm2x2(R, C, row_labels=("Cooperate", "Defect"), col_labels=("Cooperate", "Defect"))


def demo_matching_pennies() -> NormalForm2x2:
    """
    Zero-sum example with interior mixed equilibrium at (0.5, 0.5).
    """
    R = [[1, -1],
         [-1, 1]]
    C = [[-1, 1],
         [1, -1]]
    return NormalForm2x2(R, C, row_labels=("Heads", "Tails"), col_labels=("Heads", "Tails"))


def demo_market_maker_vs_trader() -> NormalForm2x2:
    """
    Toy market microstructure game:
      Row (Trader): {Aggressive, Patient}
      Column (MarketMaker): {TightSpread, WideSpread}

      Intuition:
        - If Trader is Aggressive and Maker is Tight -> Trader profits more (quick fill), Maker earns less.
        - If Trader is Patient and Maker is Wide  -> Maker safer profit, Trader less.
      Numbers are illustrative only.
    """
    R = [[1.6, 0.8],   # Trader payoff
         [0.9, 1.2]]
    C = [[0.6, 1.1],   # Maker payoff
         [1.0, 0.7]]
    return NormalForm2x2(R, C, row_labels=("Aggressive", "Patient"), col_labels=("TightSpread", "WideSpread"))


# =========================
# Pretty Printing
# =========================

def pp_game(game: NormalForm2x2) -> None:
    print(f"Rows: {game.row_labels}, Cols: {game.col_labels}")
    print("Row payoffs (R):")
    for i in range(2):
        print(f"  {game.row_labels[i]}: {game.R[i]}")
    print("Col payoffs (C):")
    for i in range(2):
        print(f"  {game.row_labels[i]}: {game.C[i]}")


# =========================
# Main (Run Demos)
# =========================

def main():
    print("\n=== Normal-form game: Matching Pennies ===")
    g = demo_matching_pennies()
    pp_game(g)

    ne = solve_mixed_nash_2x2(g)
    print(f"Mixed-strategy NE (p*, q*): {ne}  # Expected ~ (0.5, 0.5)")
    p_fp, q_fp = fictitious_play(g, iterations=5000)
    print(f"Fictitious-play empirical mix: p_row0={p_fp:.3f}, q_col0={q_fp:.3f}")

    sim = epsilon_greedy_simulation(g, epsilon=0.1, rounds=10000)
    print("ε-greedy simulation:", sim)

    print("\n=== Market Maker vs Trader (toy game) ===")
    mkt = demo_market_maker_vs_trader()
    pp_game(mkt)
    ne2 = solve_mixed_nash_2x2(mkt)
    print(f"Mixed-strategy NE (p*, q*): {ne2}")
    if ne2:
        u_r, u_c = mkt.expected_payoffs(*ne2)
        print(f"Expected payoffs at NE: Row={u_r:.3f}, Col={u_c:.3f}")

    print("\n=== Sequential Decision Tree (toy) ===")
    root = build_sample_market_tree()
    # Example policies:
    policy_bull_buy = {"MarketState": "Bullish", "BullishDecision": "Buy"}
    payoff = root.evaluate(policy_bull_buy)
    print(f"Policy {policy_bull_buy} -> terminal payoff {payoff}")

    policy_bear_hold = {"MarketState": "Bearish", "BearishDecision": "Hold"}
    payoff2 = root.evaluate(policy_bear_hold)
    print(f"Policy {policy_bear_hold} -> terminal payoff {payoff2}")

    print("\nDone. You can now tweak payoffs, add games, or export metrics to CSV/plots.")


if __name__ == "__main__":
    main()
