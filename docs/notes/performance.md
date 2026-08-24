# Performance notes

What has been measured, what was changed, and where the remaining costs live.
Development happens on an Apple M2 (CPU); local timings are indicative only — the
optimization targets are *structural* properties that carry to GPU/TPU: compile-time
scaling, vectorization shape, memory layout, and device-host round-trips.

## Where the cost lives

- **IIT 4.0** is pure tensor work. Runtime is dominated by the genuinely exponential
  mechanism-partition axis Θ(M, Z) (`ALL` scheme): the padded partition count per
  mechanism-purview pair reaches ~68k at n = 5 and ~2.2M at n = 6, so exact 4.0 at
  n = 6 is the computational frontier for any implementation (the per-class partition
  tables alone are ~15 GB at n = 6; ~180 MB at n = 5).
- **IIT 3.0** is host-bound by construction: every cause-side φ is an exact
  transportation LP. What matters on accelerators is the *number of device-host
  round-trips*, not the LP count.

## Structural changes (2026-08)

1. **Per-class scans instead of unrolled mechanism loops.** Both measures' mechanism
   drivers scanned over mechanisms of equal size (whose enumeration tables share
   shapes exactly) instead of unrolling one traced body per mechanism: one compiled
   body per size class (≤ n) rather than per mechanism (2ⁿ − 1). Measured on M2,
   `iit4.distinctions`: n = 4 compile 5.3 s → 2.1 s; n = 5 compile 28.3 s → 12.3 s and
   run 6.5 s → 4.2 s (the scan also beats the unrolled graph at runtime). The
   mechanism bitmask became a traced gather index into the master conditional tables
   to make bodies reusable across class members.
2. **One EMD callback per enumeration stage.** `iitx.distances.emd` now uses
   `vmap_method="expand_dims"`: however deeply vmapped, the whole batch crosses to the
   host in a single `pure_callback` and the LPs are solved in a loop there. Previously
   (`"sequential"`) every distance was its own host round-trip — harmless on CPU,
   ruinous on GPU.

## Known remaining costs, deliberately deferred

- **Per-candidate compilation in the complex drivers.** `iit4.complexes` and
  `iit3.major_complex` call the jitted analyses per candidate subset; candidates of
  different sizes have different cut/partition table shapes, so each size compiles
  fresh. Fix (later): pad cut tables across candidate sizes, or run the drivers
  per-size batched. On M2 the drivers are minutes for n = 5.
- **The Q² specified-state-pair table in `iit4.system_phi`.** Tie resolution
  materializes (num_cuts, Q, Q); fine through n = 5 (~5 MB), heavy at n = 6 with tens
  of thousands of cuts. Fix (later): restrict pairs to the ii-tied sets with a
  fixed-size top-k mask.
- **n = 6 exact `ALL` partitions** need streamed, on-the-fly partition generation
  (a `fori_loop` over a compact partition-index encoding rather than materialized
  tables). This is the natural boundary where the Sinkhorn/sampled approximations
  (design.md §13) take over.
- **float64.** Oracle parity runs in x64 on CPU. On GPU, run float32 for gradients and
  scale (documented tolerance envelope); TPUs have no native f64.

## Affordability on an Apple M2 (float64, CPU) — 2026-08-25

Steady-state runtimes after compilation (compile times in parentheses where they
matter); random stochastic binary systems, full-system candidate.

| n | φ_s (iit4) | ∇ soft φ_s | Φ-structure (iit4) | Φ (iit3) |
|---|---|---|---|---|
| 3 | 0.4 ms | 0.4 ms | 1 ms | 0.24 s |
| 4 | 1 ms | 1 ms | 33 ms | 7 s |
| 5 | 8 ms | 12 ms | 3.9 s (13 s compile) | 232 s cold |
| 6 | 0.53 s (10 s compile) | 1.1 s (37 s compile) | — (partition tables ~15 GB) | — |

Batched gradient throughput of the ascent objective (`vmap(grad(soft_system_phi))`):
~50,000 systems/s at n = 3 (batch 1024), ~2,800 systems/s at n = 4. Rules of thumb:

- **Interactive** (sub-second feedback): everything at n ≤ 5 except the 3.0 Φ and the
  n = 5 Φ-structure; population-scale ascent at n ≤ 4.
- **Workable** (seconds to minutes): n = 5 Φ-structures, n = 6 φ_s and its gradient
  (a 1,000-step single-system ascent ≈ 20 min), n = 4 exact 3.0 Φ.
- **Long-running**: n = 5 exact 3.0 Φ (~4 min cold; host LPs dominate), complex-search drivers at
  n = 5 (per-candidate compilation).
- **Out of reach as implemented**: n = 6 mechanism level (`ALL` partitions) and 3.0 —
  the streamed-partition frontier of the section above.
