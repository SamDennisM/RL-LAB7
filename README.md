
<p align="center">
  <img src="https://img.icons8.com/external-flatart-icons-outline-flatarticons/100/000000/artificial-intelligence.png" width="110"/>
</p>

<h1 align="center">🧠 Lab 7 — TD(0) Temporal-Difference Learning</h1>

<p align="center">
  <b>Implementation, experiments & visualizations for TD(0) (temporal-difference prediction)</b><br>
  <i>Tabular TD(0) • Monte Carlo baseline • DP reference • Stochastic transitions • Linear function approximation</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter"/>
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-ff7f0e?logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-3762A8?logo=seaborn&logoColor=white"/>
</p>

---

## 🔎 Project Overview

This repository demonstrates **TD(0)** — a temporal-difference bootstrapping algorithm for **state-value prediction** — implemented on a small **GridWorld** environment.  
The notebook includes cell-by-cell code, explanations, and many visualizations for analysis and learning.

### Core goals
- Implement **TD(0)** (tabular) for value prediction under a policy.  
- Provide **Monte Carlo (first-visit)** baseline (non-bootstrapped).  
- Compute a **reference value** using **Iterative Policy Evaluation (Dynamic Programming)**.  
- Add and evaluate complexities: **stochastic transitions (slip)** and **TD(0) with linear function approximation**.  
- Include experiments (vary α), visualizations (heatmaps, RMSE curves, visit counts), and comparisons.

---

## 🗺 Environment — GridWorld

- Default: **5 × 5** grid (customizable)  
- Start: bottom-left (0,0)  
- Terminal: top-right (4,4)  
- Step reward: **-1**; Terminal reward: **0**  
- Actions: Right, Down, Left, Up (deterministic by default)  
- Optional **slip** parameter: probability of randomizing the intended action (stochastic transitions)  
- Obstacles supported (cells agent cannot enter)

---

## ⚙️ Files (Suggested Repo Layout)

```

.
├── rl_lab7_td0.ipynb           # Full cell-by-cell Jupyter notebook (primary deliverable)
├── README.md                   # This file
└── assets/                     # (Optional) saved plots/screenshots

````

---

## ▶️ How to run (local / Colab)

1. Clone repository:
```bash
git clone https://github.com/your-username/rl-td0-lab7.git
cd rl-td0-lab7
````

2. Install dependencies:

```bash
pip install numpy matplotlib seaborn jupyter
```

3. Open the notebook:

```bash
jupyter notebook rl_lab7_td0.ipynb
```

Or upload to Google Colab and run each cell.

---

## 🧩 Notebook Contents (Cell-by-cell summary)

1. **Title & Goals** — brief description and objectives.
2. **Imports** — `numpy`, `matplotlib`, `seaborn`, `collections.defaultdict`, etc.
3. **GridWorld class** — deterministic & stochastic versions, rendering utilities for heatmaps and policy display.
4. **Policy** — uniform random policy implementation used for prediction.
5. **Iterative Policy Evaluation (DP)** — compute reference "true" value under the policy.
6. **TD(0) (tabular)** — step-by-step implementation, RMSE tracking vs DP, state-visit heatmap.
7. **Monte Carlo (first-visit)** — baseline estimator; compare RMSE with TD(0).
8. **Experiment: Learning rate (α) sweep** — runs TD(0) for different α values and plots RMSE curves.
9. **Added Complexity 1: Stochastic transitions** — enable slip, recompute DP reference, run TD(0), visualize & compare.
10. **Added Complexity 2: TD(0) with Linear Function Approximation** — featurize states and learn weights with TD updates; show RMSE & approximated value heatmap.
11. **Visualizations** — value heatmaps, RMSE curves, visit counts, weight evolution plots.
12. **Summary & notes** — instructions to tweak hyperparameters and further experiments.

---

## 📊 Visualizations Included

* DP reference value heatmap (Iterative Policy Evaluation)
* TD(0) estimated value heatmap (tabular)
* Monte Carlo estimated value heatmap
* RMSE vs DP over episodes (TD(0) & MC)
* Learning curves for different α values
* State visit counts heatmap during TD learning
* RMSE and weight evolution for linear function approximation
* Comparison visualizations for deterministic vs stochastic environments

---

## ✅ Additional Complexity (explained)

1. **Stochastic Transitions (Slip)**

   * Add `slip` probability: with probability `slip` the agent takes a random action instead of the intended one.
   * Purpose: study TD(0) robustness when dynamics are non-deterministic; recompute DP reference to account for stochasticity.

2. **TD(0) with Linear Function Approximation**

   * Feature examples: normalized coordinates and interaction terms, e.g.
     `φ(s) = [x/(W-1), y/(H-1), (x*y)/((W-1)*(H-1)), 1]`
   * Update rule (per TD step):
     `w ← w + α (r + γ wᵀφ(s') − wᵀφ(s)) φ(s)`
   * Purpose: demonstrate generalization, weight learning, and compare approximation vs tabular TD.

---

## 🔬 Suggested Experiments (try these in the notebook)

* Sweep λ / α / γ and plot convergence statistics.
* Increase `slip` to 0.3–0.5 and observe RMSE changes.
* Add obstacles and measure state visit redistribution.
* Replace random policy with a fixed suboptimal policy to study value differences.
* Extend to n-step TD or eligibility traces (Sarsa(λ)).
* Port to OpenAI Gym small domains (e.g., `FrozenLake-v1`) for comparison.

---

## 📚 Notes & References

* Sutton & Barto, *Reinforcement Learning: An Introduction* — Ch. 6 (Temporal-Difference Learning)
* TD(0) is a one-step bootstrapping method (mix of Monte Carlo and DP).
* Use DP reference only for small MDPs (state space must be tractable).

---

## 🏁 Final Remarks

This notebook is intentionally self-contained and reproducible. It provides:

* A clear, single-place implementation of TD(0) with comparisons and visual diagnostics.
* Two practical complexity additions (stochastic dynamics and function approximation) that make the lab suitable for deeper analysis and assignments.

---

<p align="center">Made with ❤️ · Happy experimenting!</p>
```
