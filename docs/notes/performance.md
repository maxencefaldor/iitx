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
