# Experiments — gradients of Φ

For the first time, Φ has a gradient. This directory is the lab where we find out what
that makes possible. It is research, not documentation: notebooks live here until one
matures into something curated, at which point it is promoted to `examples/` and the
docs. The background and literature review are in
[`docs/notes/phi-optimization.md`](../docs/notes/phi-optimization.md).

## Rules

1. **The prediction comes before the run.** Every notebook opens with pre-registered
   predictions, written before the first execution. The verdict at the end says which
   held.
2. **Only exact values are called Φ.** Relaxation values are always labeled as such.
   Any experiment that ascends a surrogate also tracks the exact quantity, so a
   relaxation that climbs while the exact value does not can never fail silently.
3. **Distributions, never maxima.** Multi-seed always; report histograms and quantiles.
   A best-of-N number appears only next to its N.
4. **The executed notebook is the run log.** Notebooks are committed executed; the
   first cell prints the `iitx` version and every seed. Large arrays go to
   `experiments/data/` (gitignored) and are regenerable by re-running the notebook.
5. **Self-contained notebooks.** Each notebook redefines its own small helpers, even
   when that duplicates another notebook — a stranger must be able to read one file
   alone. A helper graduates to `src/iitx` only when it is theory-level (a quantity
   with a citation, not experiment plumbing) and more than one notebook needs it.
6. **Exploits are results.** Every striking number is treated as a possible exploit of
   the definition until shown otherwise.
7. Stray ideas get a file in [`notes/ideas/`](notes/ideas/); every gotcha gets one in
   [`notes/friction/`](notes/friction/); dead ends are recorded with their cause of
   death.

## Index

| Notebook | Question | Status |
|---|---|---|
| [`e00_gradient_audit.ipynb`](e00_gradient_audit.ipynb) | Which ascent objective can we trust — exact subgradient, soft, or annealed — and where do they disagree? | done |
| [`e01_n3_atlas.ipynb`](e01_n3_atlas.ipynb) | The complete φ_s atlas of 3-unit binary systems: what does the theory reward, and does gradient ascent find it? | done |
| [`e02_exploit_survival.ipynb`](e02_exploit_survival.ipynb) | Does the frozen-state exploit survive the 2026 cap, reachability weighting, and scaling in n — and what are ascent's needles really? | done |
| [`e03_one_ibit_ceiling.ipynb`](e03_one_ibit_ceiling.ipynb) | Why 2026 ascent crowds 1 ibit at n = 3: the cause-surprisal crossing, the attained 1.000000, and the ≈0.6·log₂ n growth of the 2026 ceiling | done |
| [`e04_unfolding_descent.ipynb`](e04_unfolding_descent.ipynb) | Unfolding by gradient: 90% of φ_s removed at 0.5% behavior error within a fixed architecture — with a ~1-ibit floor, unchanged wiring, and big Φ *rising* as φ_s collapses | done |
| [`e05_ising_criticality.ipynb`](e05_ising_criticality.ipynb) | Ising rings, exact E_π[φ]: 2023 climbs monotonically into the ordered phase (no critical peak — contra the proxy literature), 2026 peaks at finite noise below the crossover | done |
| [`e06_ascent_phenomenology.ipynb`](e06_ascent_phenomenology.ipynb) | Watching ascent at n = 4: degeneracy rises 86× (anti-specialization), MIP switches are the kinks, the maximin tie-ridge fails to appear, and big Φ doubles while *losing* distinctions | done |
