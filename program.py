from __future__ import annotations

import os
import re
import sys
import csv
import random
from dataclasses import dataclass
from collections import Counter
from itertools import combinations
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


SEED = 15
ROUNDS_DEFAULT = 100
TRIALS_DEFAULT = 10_000
TILES = ["A", "B", "C", "D"]

@dataclass(frozen=True)
class Ruleset:
    use_rule1_pairs: bool = True  
    use_rule2_prev_singleton: bool = True 
    use_rule3_all_same_override: bool = True 

COLOR = {
    "Anti Copier": "#FF0000",       
    "Random 1": "#8EC5FF",         
    "Random 2": "#64B5F6",         
    "Random 3": "#A8D1FF",      
    "Self Copier": "#2ECC71",   
    "Leader Copier": "#FFA500",     
    "Frequency Copier": "#8A2BE2",   
}
EDGE = "#0B0B0B"


def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:180] if len(s) > 180 else s


def print_environment_info() -> None:
    import platform
    print("=== Environment Info (for reproducibility) ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("Matplotlib:", plt.matplotlib.__version__)
    print("Base seed:", SEED)
    print("Figures folder:", FIGURES_DIR)
    print("============================================")


def reset_rng(seed: int) -> random.Random:
    return random.Random(seed)


def mean_ci_95(x: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 0 else 0.0
    half = 1.96 * se
    return mean, sd, mean - half, mean + half


def format_pm(mean: float, ci_low: float, ci_high: float, decimals: int = 1) -> str:
    half = (ci_high - ci_low) / 2.0
    return f"{mean:.{decimals}f} ± {half:.{decimals}f}"

def bootstrap_delta_mean_ci(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size

    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        sa = rng.choice(a, size=n_a, replace=True)
        sb = rng.choice(b, size=n_b, replace=True)
        deltas[i] = sa.mean() - sb.mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_samples: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)

    if a.size > max_samples:
        a = rng.choice(a, size=max_samples, replace=False)
    if b.size > max_samples:
        b = rng.choice(b, size=max_samples, replace=False)

    diff = a[:, None] - b[None, :]
    wins = np.sum(diff > 0)
    losses = np.sum(diff < 0)
    n = diff.size
    return float((wins - losses) / n)

import math

def wilson_ci_95_from_wins(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def sim_type_from_id(sim_id: int) -> str:
    if 1 <= sim_id <= 4:
        return "1v3"
    if 5 <= sim_id <= 10:
        return "2v2"
    if 11 <= sim_id <= 14:
        return "3v1"
    if sim_id == 15:
        return "4v0"
    return "?"



def score_round(choices: List[str],
                prev_choices: Optional[List[str]],
                ruleset: Ruleset) -> List[int]:
    n = len(choices)
    pts = [0] * n

    if ruleset.use_rule3_all_same_override and len(set(choices)) == 1:
        return pts

    if ruleset.use_rule1_pairs:
        counts_now = Counter(choices)
        for tile, c in counts_now.items():
            if c == 2:
                for i, ch in enumerate(choices):
                    if ch == tile:
                        pts[i] += 1

    if ruleset.use_rule2_prev_singleton and prev_choices is not None and len(prev_choices) > 0:
        counts_prev = Counter(prev_choices)
        for i, ch in enumerate(choices):
            if counts_prev.get(ch, 0) == 1:
                pts[i] += 1

    return pts


@dataclass
class Agent:
    name: str
    rng: random.Random

    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        raise NotImplementedError


@dataclass
class RandomAgent(Agent):
    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        return self.rng.choice(TILES)


@dataclass
class SelfCopier(Agent):
    locked_tile: Optional[str] = None

    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        if self.locked_tile is not None:
            return self.locked_tile
        return self.rng.choice(TILES)

    def maybe_lock(self, chosen_tile: str, points_earned_this_round: int) -> None:
        if self.locked_tile is None and points_earned_this_round > 0:
            self.locked_tile = chosen_tile


@dataclass
class LeaderCopier(Agent):
    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        if len(history[i]) == 0:
            return self.rng.choice(TILES)

        max_score = max(scores)
        leaders = [idx for idx, s in enumerate(scores) if s == max_score] 
        leader = self.rng.choice(leaders)

        if len(history[leader]) == 0:
            return self.rng.choice(TILES)
        return history[leader][-1]


@dataclass
class FrequencyCopier(Agent):
    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        if len(history[i]) == 0:
            return self.rng.choice(TILES)

        last_round = [h[-1] for h in history if len(h) > 0]
        if not last_round:
            return self.rng.choice(TILES)

        counts = Counter(last_round)
        max_freq = max(counts.values())
        top = [tile for tile, c in counts.items() if c == max_freq]
        return self.rng.choice(top)


@dataclass
class AntiCopier(Agent):
    def choose(self, history: List[List[str]], scores: List[int], i: int) -> str:
        if len(history[i]) == 0:
            return self.rng.choice(TILES)

        last_round = [h[-1] for h in history if len(h) > 0]
        if not last_round:
            return self.rng.choice(TILES)

        counts = Counter(last_round)
        min_freq = min(counts.values())
        least = [tile for tile, c in counts.items() if c == min_freq]
        return self.rng.choice(least)


def make_agent(name: str, rng: random.Random, index_for_random: int = 1) -> Agent:
    if name.startswith("Random"):
        return RandomAgent(name=f"Random {index_for_random}", rng=rng)
    if name == "Self Copier":
        return SelfCopier(name=name, rng=rng)
    if name == "Leader Copier":
        return LeaderCopier(name=name, rng=rng)
    if name == "Frequency Copier":
        return FrequencyCopier(name=name, rng=rng)
    if name == "Anti Copier":
        return AntiCopier(name=name, rng=rng)
    raise ValueError(f"Unknown agent name: {name}")


EXPERIMENTAL = ["Leader Copier", "Self Copier", "Frequency Copier", "Anti Copier"]


def run_game_round_scores(agent_objs: List[Agent], rounds: int, ruleset: Ruleset) -> List[List[int]]:
    n = len(agent_objs)
    history = [[] for _ in range(n)]
    total = [0] * n
    prev_choices: Optional[List[str]] = None
    per_round = [[] for _ in range(n)]

    for _ in range(rounds):
        choices = [agent_objs[i].choose(history, total, i) for i in range(n)]
        pts = score_round(choices, prev_choices, ruleset)

        for i in range(n):
            history[i].append(choices[i])
            total[i] += pts[i]
            per_round[i].append(pts[i])

            if isinstance(agent_objs[i], SelfCopier):
                agent_objs[i].maybe_lock(choices[i], pts[i])

        prev_choices = choices[:]

    return per_round


def pooled_running_avg(trials_x_rounds: np.ndarray) -> np.ndarray:
    cums = np.cumsum(trials_x_rounds, axis=1)
    denom = np.arange(1, trials_x_rounds.shape[1] + 1)[None, :]
    per_trial_running = cums / denom
    return per_trial_running.mean(axis=0)


def find_convergence(mu: np.ndarray, window: int = 20, eps: float = 0.01, checks: int = 5) -> Optional[int]:
    R = len(mu)
    if R < window + checks:
        return None

    ok = 0
    for t in range(window, R):
        if abs(mu[t] - mu[t - window]) < eps:
            ok += 1
            if ok >= checks:
                return (t - checks + 2)
        else:
            ok = 0
    return None

CONV_GRID = [
    (10, 0.005, 3),
    (10, 0.010, 3),
    (10, 0.020, 3),
    (20, 0.005, 5),
    (20, 0.010, 5), 
    (20, 0.020, 5),
    (30, 0.005, 7),
    (30, 0.010, 7),
    (30, 0.020, 7),
]

def convergence_sensitivity(mu_dict: Dict[str, np.ndarray],
                            grid=CONV_GRID) -> Dict[str, Dict[Tuple[int, float, int], Optional[int]]]:
    out: Dict[str, Dict[Tuple[int, float, int], Optional[int]]] = {k: {} for k in mu_dict.keys()}
    for (w, eps, c) in grid:
        for label, mu in mu_dict.items():
            out[label][(w, eps, c)] = find_convergence(mu, window=w, eps=eps, checks=c)
    return out

def summarize_convergence_robustness(sens: Dict[str, Dict[Tuple[int, float, int], Optional[int]]]
                                   ) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for label, d in sens.items():
        vals = [r for r in d.values() if r is not None]
        frac = len(vals) / max(1, len(d))
        if len(vals) == 0:
            summary[label] = {
                "frac_converged": frac,
                "median_round": float("nan"),
                "min_round": float("nan"),
                "max_round": float("nan"),
            }
        else:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            median = float(vals_sorted[mid]) if len(vals_sorted) % 2 == 1 else 0.5 * (vals_sorted[mid-1] + vals_sorted[mid])
            summary[label] = {
                "frac_converged": frac,
                "median_round": median,
                "min_round": float(min(vals_sorted)),
                "max_round": float(max(vals_sorted)),
            }
    return summary


def plot_distribution(scores_dict: Dict[str, List[int]],
                      mean_ci_dict: Dict[str, Tuple[float, float, float, float]],
                      win_ci_dict: Dict[str, Tuple[float, float, float, float]],
                      title: str,
                      filename: str) -> None:
    pooled = [s for arr in scores_dict.values() for s in arr]
    bins = range(min(pooled), max(pooled) + 2, 2)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    keys = list(scores_dict.keys())
    random_keys = [k for k in keys if k.startswith("Random")]
    strat_keys = [k for k in keys if not k.startswith("Random")]

    for k in random_keys:
        arr = scores_dict[k]
        c = COLOR.get(k, "#8EC5FF")

        mean, sd, lo, hi = mean_ci_dict[k]
        w_mean, w_sd, w_lo, w_hi = win_ci_dict[k]
        lab = f"{k} (μ={format_pm(mean, lo, hi, 1)}, win={format_pm(100*w_mean, 100*w_lo, 100*w_hi, 1)}%)"

        plt.hist(arr, bins=bins, histtype="step", linewidth=4.0, color=c, label=lab, zorder=1)
        plt.axvline(mean, color=c, linestyle="--", linewidth=2.0, alpha=0.95, zorder=2)

    for k in strat_keys:
        arr = scores_dict[k]
        c = COLOR.get(k, "#999999")

        mean, sd, lo, hi = mean_ci_dict[k]
        w_mean, w_sd, w_lo, w_hi = win_ci_dict[k]
        lab = f"{k} (μ={format_pm(mean, lo, hi, 1)}, win={format_pm(100*w_mean, 100*w_lo, 100*w_hi, 1)}%)"

        plt.hist(arr, bins=bins, alpha=0.60, color=c, edgecolor=EDGE, linewidth=0.0, label=lab, zorder=3)
        plt.axvline(mean, color=c, linestyle="-", linewidth=2.6, alpha=0.98, zorder=4)

    plt.xlabel("Final Score After 100 Rounds")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#999", fontsize=8)
    plt.tight_layout()

    outpath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence(mu_dict: Dict[str, np.ndarray],
                     title: str,
                     conv_rounds: Dict[str, Optional[int]],
                     filename: str) -> None:
    rounds = len(next(iter(mu_dict.values())))
    x = np.arange(1, rounds + 1)

    plt.figure(figsize=(12, 5.6))
    for label, mu in mu_dict.items():
        c = COLOR.get(label, "#999999")
        plt.plot(x, mu, color=c, linewidth=2.3, alpha=0.95, label=label)

    for label, r in conv_rounds.items():
        if r is not None:
            c = COLOR.get(label, "#999999")
            plt.axvline(r, color=c, linestyle="--", linewidth=1.6, alpha=0.85)

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Pooled running average score")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    outpath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def simulate_distribution_and_convergence(
    roster_names: List[str],
    sim_id: int,
    trials: int,
    rounds: int,
    ruleset: Ruleset,
    save_plots: bool = True,
) -> Dict[str, object]:
    base_rng = reset_rng(SEED + sim_id) 

    labels: List[str] = []
    rcount = 0
    for nm in roster_names:
        if nm == "Random":
            rcount += 1
            labels.append(f"Random {rcount}")
        else:
            labels.append(nm)

    final_scores: Dict[str, List[int]] = {lab: [] for lab in labels}
    win_share: Dict[str, np.ndarray] = {lab: np.zeros(trials, dtype=np.float64) for lab in labels}
    conv_points: Dict[str, np.ndarray] = {lab: np.zeros((trials, rounds), dtype=np.float32) for lab in labels}

    for t in range(trials):
        trial_seeds = [base_rng.getrandbits(64) for _ in range(4)]
        trial_rngs = [random.Random(s) for s in trial_seeds]

        agent_objs: List[Agent] = []
        r_idx = 0
        for i, nm in enumerate(roster_names):
            if nm == "Random":
                r_idx += 1
                agent_objs.append(make_agent("Random", trial_rngs[i], index_for_random=r_idx))
            else:
                agent_objs.append(make_agent(nm, trial_rngs[i]))

        per_round = run_game_round_scores(agent_objs, rounds=rounds, ruleset=ruleset)
        finals = [int(sum(per_round[i])) for i in range(4)]

        for i, lab in enumerate(labels):
            final_scores[lab].append(finals[i])
            conv_points[lab][t, :] = np.array(per_round[i], dtype=np.float32)

        m = max(finals)
        winners = [i for i, s in enumerate(finals) if s == m]
        share = 1.0 / len(winners)
        for i in winners:
            win_share[labels[i]][t] += share

    mean_ci_dict: Dict[str, Tuple[float, float, float, float]] = {}
    for lab in labels:
        arr = np.array(final_scores[lab], dtype=np.float64)
        mean_ci_dict[lab] = mean_ci_95(arr)  

    win_ci_dict: Dict[str, Tuple[float, float, float, float]] = {}
    for lab in labels:
        wins = float(win_share[lab].sum()) 
        p_hat = wins / trials         
        lo, hi = wilson_ci_95_from_wins(wins, trials)
        win_ci_dict[lab] = (p_hat, 0.0, lo, hi)

    
    random_labels = [lab for lab in labels if lab.startswith("Random")]
    baseline_scores = None
    if len(random_labels) > 0:
        baseline_scores = np.concatenate([np.array(final_scores[rl], dtype=np.float64) for rl in random_labels])
    
    practical: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        arr = np.array(final_scores[lab], dtype=np.float64)

        if baseline_scores is None:
            practical[lab] = {
                "baseline_mean": float("nan"),
                "delta_mean": float("nan"),
                "delta_ci_low": float("nan"),
                "delta_ci_high": float("nan"),
                "delta_per_round": float("nan"),
                "cliffs_delta": float("nan"),
            }
            continue

        base_mean = float(baseline_scores.mean())
        delta_mean = float(arr.mean() - base_mean)
        d_lo, d_hi = bootstrap_delta_mean_ci(arr, baseline_scores, reps=2000, seed=SEED + 1000 * sim_id)
        cd = cliffs_delta(arr, baseline_scores, max_samples=2000, seed=SEED + 2000 * sim_id)

        practical[lab] = {
            "baseline_mean": base_mean,
            "delta_mean": delta_mean,
            "delta_ci_low": d_lo,
            "delta_ci_high": d_hi,
            "delta_per_round": delta_mean / float(rounds),
            "cliffs_delta": cd,
        }

    mu_dict = {lab: pooled_running_avg(conv_points[lab]) for lab in labels}

    conv_rounds = {lab: find_convergence(mu_dict[lab]) for lab in labels}

    conv_sens = convergence_sensitivity(mu_dict, grid=CONV_GRID)
    conv_robust = summarize_convergence_robustness(conv_sens)


    files_out = {"distribution": "", "convergence": ""}

    if save_plots:
        title_dist = f"Simulation {sim_id} — Distribution"
        fname_dist = f"sim_{sim_id:02d}_distribution_{sanitize_filename('_'.join(labels))}.png"
        plot_distribution(final_scores, mean_ci_dict, win_ci_dict, title_dist, fname_dist)

        title_conv = f"Simulation {sim_id} — Convergence"
        fname_conv = f"sim_{sim_id:02d}_convergence_{sanitize_filename('_'.join(labels))}.png"
        plot_convergence(mu_dict, title_conv, conv_rounds, fname_conv)

        files_out = {
            "distribution": os.path.join(FIGURES_DIR, fname_dist),
            "convergence": os.path.join(FIGURES_DIR, fname_conv),
        }

    return {
        "sim_id": sim_id,
        "roster": roster_names,
        "labels": labels,
        "score_stats": mean_ci_dict,    
        "win_stats": win_ci_dict,     
        "convergence_round": conv_rounds,
        "practical": practical,
        "convergence_sensitivity": conv_sens,
        "convergence_robustness": conv_robust,
        "files": files_out,
    }


def build_simulation_rosters() -> List[Tuple[int, List[str]]]:
    rosters: List[Tuple[int, List[str]]] = []
    sim_id = 1

    
    for exp in EXPERIMENTAL:
        rosters.append((sim_id, ["Random", "Random", "Random", exp]))
        sim_id += 1

    
    for a, b in combinations(EXPERIMENTAL, 2):
        rosters.append((sim_id, ["Random", "Random", a, b]))
        sim_id += 1

    
    for trio in combinations(EXPERIMENTAL, 3):
        rosters.append((sim_id, [trio[0], trio[1], trio[2], "Random"]))
        sim_id += 1

    
    rosters.append((sim_id, [EXPERIMENTAL[0], EXPERIMENTAL[1], EXPERIMENTAL[2], EXPERIMENTAL[3]]))
    return rosters


if __name__ == "__main__":
    print_environment_info()

    trials = TRIALS_DEFAULT
    rounds = ROUNDS_DEFAULT
    ruleset = Ruleset(True, True, True) 

    rosters = build_simulation_rosters()

    print(f"Running {len(rosters)} simulations with trials={trials}, rounds={rounds} ...")
    print("Saving figures to:", FIGURES_DIR)
    print()

    csv_path = os.path.join(FIGURES_DIR, "summary_simulations.csv")
    csv_sens_path = os.path.join(FIGURES_DIR, "convergence_sensitivity.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "simulation_id",
            "sim_type",
            "roster",
            "agent",
            "mean_score",
            "ci_halfwidth_score",
            "win_rate_percent",
            "ci_halfwidth_win_rate_percent",
            "convergence_round",
            "baseline_mean_random",
            "delta_vs_random_mean",
            "delta_vs_random_ci_low",
            "delta_vs_random_ci_high",
            "delta_vs_random_per_round",
            "cliffs_delta_vs_random",
        ])

        fsens = open(csv_sens_path, "w", newline="")
        wsens = csv.writer(fsens)
        wsens.writerow([
            "simulation_id",
            "sim_type",
            "roster",
            "agent",
            "w",
            "eps",
            "c",
            "convergence_round",
        ])

        for sim_id, roster in rosters:
            res = simulate_distribution_and_convergence(
                roster,
                sim_id=sim_id,
                trials=trials,
                rounds=rounds,
                ruleset=ruleset,
            )

            conv_sens = res["convergence_sensitivity"]
            labels = res["labels"]
            score_stats = res["score_stats"]   
            win_stats = res["win_stats"]      
            conv = res["convergence_round"]
            practical = res["practical"]
            sim_type = sim_type_from_id(sim_id)

            for lab in labels:
                for (w, eps, c), r in conv_sens[lab].items():
                    wsens.writerow([
                        sim_id,
                        sim_type,
                        " | ".join(res["roster"]),
                        lab,
                        w,
                        eps,
                        c,
                        r if r is not None else "",
                    ])

            print(f"Sim {sim_id:02d}: {labels}")
            for lab in labels:
                m, _sd, lo, hi = score_stats[lab]
                wm, _wsd, wlo, whi = win_stats[lab]

                score_ci_half = (hi - lo) / 2.0
                win_rate_percent = 100.0 * wm
                win_ci_half_percent = 100.0 * ((whi - wlo) / 2.0)

                mean_pm = f"{m:.1f} ± {score_ci_half:.1f}"
                win_pm = f"{win_rate_percent:.1f}% ± {win_ci_half_percent:.1f}%"
                print(f"  {lab:14s}  mean={mean_pm:>14s}   win={win_pm:>18s}   conv={conv[lab]}")

                pr = practical[lab]
                writer.writerow([
                    sim_id,
                    sim_type,
                    " | ".join(res["roster"]),
                    lab,
                    m,
                    score_ci_half,
                    win_rate_percent,
                    win_ci_half_percent,
                    conv[lab] if conv[lab] is not None else "",
                    pr["baseline_mean"],
                    pr["delta_mean"],
                    pr["delta_ci_low"],
                    pr["delta_ci_high"],
                    pr["delta_per_round"],
                    pr["cliffs_delta"],
                ])

            print()

        fsens.close()

    ablations = [
        ("baseline", Ruleset(True, True, True)),
        ("no_rule1_pairs", Ruleset(False, True, True)),
        ("no_rule2_prev_singleton", Ruleset(True, False, True)),
        ("no_rule3_all_same_override", Ruleset(True, True, False)),
    ]

    ablation_csv_path = os.path.join(FIGURES_DIR, "summary_rule_ablations.csv")
    ablation_sens_path = os.path.join(FIGURES_DIR, "convergence_sensitivity_rule_ablations.csv")

    with open(ablation_csv_path, "w", newline="") as f2:
        writer2 = csv.writer(f2)
        writer2.writerow([
            "ruleset",
            "simulation_id",
            "sim_type",
            "roster",
            "agent",
            "mean_score",
            "ci_halfwidth_score",
            "win_rate_percent",
            "ci_halfwidth_win_rate_percent",
            "convergence_round",
            "baseline_mean_random",
            "delta_vs_random_mean",
            "delta_vs_random_ci_low",
            "delta_vs_random_ci_high",
            "delta_vs_random_per_round",
            "cliffs_delta_vs_random",
        ])

        f2sens = open(ablation_sens_path, "w", newline="")
        w2sens = csv.writer(f2sens)
        w2sens.writerow([
            "ruleset",
            "simulation_id",
            "sim_type",
            "roster",
            "agent",
            "w",
            "eps",
            "c",
            "convergence_round",
        ])
        
        baseline_cache = {}
        for sim_id, roster in rosters:
            baseline_cache[sim_id] = simulate_distribution_and_convergence(
                roster, sim_id=sim_id, trials=trials, rounds=rounds,
                ruleset=Ruleset(True, True, True),
                save_plots=False,
            )
            

        for ruleset_name, ruleset_ab in ablations:
            print(f"=== Running rule ablation: {ruleset_name} ===")

            for sim_id, roster in rosters:
                res = simulate_distribution_and_convergence(
                    roster,
                    sim_id=sim_id,
                    trials=trials,
                    rounds=rounds,
                    ruleset=ruleset_ab,
                    save_plots=False,
                )
                
                

                sim_type = sim_type_from_id(sim_id)
                labels = res["labels"]
                score_stats = res["score_stats"]
                win_stats = res["win_stats"]
                conv = res["convergence_round"]
                practical = res["practical"]
                conv_sens = res["convergence_sensitivity"]

                for lab in labels:
                    for (w, eps, c), r in conv_sens[lab].items():
                        w2sens.writerow([
                            ruleset_name,
                            sim_id,
                            sim_type,
                            " | ".join(res["roster"]),
                            lab,
                            w,
                            eps,
                            c,
                            r if r is not None else "",
                        ])

                base = baseline_cache[sim_id]

                for lab in labels:
                    m, _sd, lo, hi = score_stats[lab]
                    wm, _wsd, wlo, whi = win_stats[lab]

                    score_ci_half = (hi - lo) / 2.0
                    win_rate_percent = 100.0 * wm
                    win_ci_half_percent = 100.0 * ((whi - wlo) / 2.0)

                    

                    pr = practical[lab]
                    writer2.writerow([
                        ruleset_name,
                        sim_id,
                        sim_type,
                        " | ".join(res["roster"]),
                        lab,
                        m,
                        score_ci_half,
                        win_rate_percent,
                        win_ci_half_percent,
                        conv[lab] if conv[lab] is not None else "",
                        pr["baseline_mean"],
                        pr["delta_mean"],
                        pr["delta_ci_low"],
                        pr["delta_ci_high"],
                        pr["delta_per_round"],
                        pr["cliffs_delta"],
                    ])

            print()

        f2sens.close()

    print("DONE.")
    print("All figures saved to:", FIGURES_DIR)
    print("Main CSV summary saved to:", csv_path)
    print("Main convergence sensitivity CSV saved to:", csv_sens_path)
    print("Rule ablation CSV saved to:", ablation_csv_path)
    print("Rule ablation convergence sensitivity CSV saved to:", ablation_sens_path)
