# iitx glossary

One vocabulary, used identically in code, docstrings, documentation, and tests. Where
the literature is consistent, names match the literature; where the papers disagree
(3.0 vs 4.0), the core uses version-neutral terms and each measure module uses its own
paper's terms. Symbols follow the papers; Python names follow the rules at the end.

## Core (version-neutral)

| Term | Symbol | Definition |
|---|---|---|
| unit | $U_i$ | One variable of the system, with a finite alphabet of $q_i$ states. Never "node" or "element". |
| alphabet | $\Omega_{U_i}$ | The states a unit can take; its size is $q_i$. |
| shape | — | The tuple $(q_1, …, q_n)$ of per-unit alphabet sizes; equals the tensor shape of any full state-space array. |
| state | $u$, $s$, $m$, $z$ | An assignment of a value to each unit of some set, as an `(n,)` integer vector; lowercase of the set's symbol. |
| state index | — | The little-endian mixed-radix flat index of a state (unit 0 is the fastest-varying digit). |
| TPM | $\mathcal{T}$ | Transition probability matrix $p(\bar u \mid do(u))$, state-by-state `(Q, Q)`, rows = prior state; interventional, rows sum to 1. |
| node TPMs | — | The factored view: per-unit conditionals $p(\bar u_i \mid u)$, shape `(*shape, q_i)`; exists iff the TPM is conditionally independent. |
| conditional independence | — | $p(\bar u \mid u) = \prod_i p(\bar u_i \mid u)$. Required by mechanism-level analysis; a checked property in iitx, never an assumption. |
| connectivity matrix | `cm` | `(n, n)` boolean; `cm[i, j] = 1` ⇔ unit $i$ inputs to unit $j$. |
| system | $U$, $S$ | The object under analysis: TPM + shape + connectivity (`System`). Never "network" (that's a provenance) or "substrate" (4.0-specific honorific). |
| candidate system | $S \subseteq U$ | A subset of units analyzed as a system, given as a mask; the rest is background. |
| background | $W = U \setminus S$ | Units outside the candidate system; conditioned per the measure's convention (frozen for 3.0; causally marginalized on the cause side for 4.0). |
| mechanism | $M$, state $m$ | A subset of the candidate system, in its current state, whose causal powers are assessed. |
| purview | $Z$, state $z$ | A subset of the candidate system over which a mechanism's cause or effect is evaluated. |
| direction | — | `Direction.CAUSE` (about $t-1$) or `Direction.EFFECT` (about $t+1$). Every two-sided quantity is one function of a `Direction`. |
| repertoire | $p(z \mid m)$, $\pi(z \mid m)$ | The distribution a mechanism specifies over a purview, in the given direction. |
| unconstrained repertoire | $p^{uc}$, $\pi(z; M)$ | The same distribution with the mechanism unconstrained (uniformly averaged out). |
| partition (mechanism) | $\theta \in \Theta(M, Z)$ | A way of severing the mechanism–purview dependency; its repertoire is the product over parts. |
| partition (system) / cut | $\theta \in \Theta(S)$ | A way of severing the candidate system; cut connections are noised (uniformly marginalized inputs). |
| MIP | $\theta'$ | Minimum (information) partition: the partition that makes the least difference, per the measure's normalization and tie policy. |
| phi | $\varphi$ | Integrated information of a mechanism-level or system-level object: the difference the MIP makes. Never bare `phi` in the public API — always a qualified name (below). |
| complex | $S^*$ | A candidate system that is a maximum of the measure's system quantity among overlapping candidates. |
| major complex / first complex | — | The complex with the globally maximal value (3.0 / 4.0 naming respectively). |
| condensation | — | The disjoint, exhaustive set of complexes of a universe. |
| precision | — | The quantization applied to every value before comparison; ties are defined at this precision (1e-6 for IIT3, 1e-13 for IIT4). |
| tie policy | — | The documented composite key and canonical order that make every selection deterministic. |
| measure | — | A first-class object mapping `(System, state)` to integrated-information results (`Measure` protocol); `IIT3` and `IIT4` are instances. |

## IIT 3.0 terms (module `iitx.measures.iit3`)

| Term | Symbol | Definition |
|---|---|---|
| cause/effect information | $ci$, $ei$ | EMD between a repertoire and its unconstrained counterpart. |
| small phi | $\varphi$ | $\min(\varphi_{\text{cause}}, \varphi_{\text{effect}})$ at the mechanism MIP, EMD-valued. |
| core cause / core effect | — | The purview maximizing $\varphi$ on each side. |
| MICE | — | Maximally irreducible cause-effect: the pair of core repertoires. |
| concept | — | A mechanism with $\varphi^{\max} > 0$, its MICE, and its $\varphi^{\max}$. |
| cause-effect structure (CES) | $C$ | The set of all concepts of a candidate system ("constellation"). |
| conceptual information | $CI$ | $\sum_c \varphi^{\max}(c)\, d(c, p^{uc})$. |
| null concept | $p^{uc}$ | The unconstrained cause-effect repertoire; the sink for destroyed concepts in XEMD. |
| XEMD | — | Extended EMD between two cause-effect structures ($\varphi$ mass transported at concept-distance cost). |
| big phi | $\Phi$ | XEMD between the CES and the CES under the system MIP cut. The headline scalar of IIT3. |

## IIT 4.0 terms (module `iitx.measures.iit4`)

| Term | Symbol | Definition |
|---|---|---|
| intrinsic difference | ID | $\max_\alpha p_\alpha \log_2(p_\alpha / q_\alpha)$; the unique measure satisfying causality, intrinsicality, specificity. Unit: ibit. |
| selectivity / informativeness | — | The $p_\alpha$ factor / the $\log_2(p_\alpha/q_\alpha)$ factor of ID-style quantities. |
| intrinsic information | $ii$ | Selectivity × informativeness of a state, against the unconstrained probability (Eqs. 5, 7, 34, 35). |
| specified state | $s'$, $z'$, $z^*$ | The state maximizing $ii$ (system: maximal cause-effect state $s' = \{s'_c, s'_e\}$; mechanism: per-purview $z'$, over purviews $z^*$). |
| backward probability | $p_c^{\leftarrow}$ | Bayes inversion with uniform prior (cause-side selectivity). |
| system phi | $\varphi_s$ | $\min(\varphi_c, \varphi_e)$ over the normalized-MIP system partition. The headline scalar of IIT4 existence; may be negative (negative ⇒ reducible). |
| distinction | $d(m)$ | A mechanism with its specified cause/effect purview-states and $\varphi_d > 0$, congruent with $s'$. |
| distinction phi | $\varphi_d$ | $\min(\varphi_c(m), \varphi_e(m))$ over disintegrating partitions, max over purviews. |
| congruence | — | $z^*_c \subseteq s'_c$ and $z^*_e \subseteq s'_e$ as unit–state sets; incongruent candidates do not exist. |
| relation | $r(\mathbf d)$ | Congruent overlap among the purviews of a set of distinctions. |
| relation phi | $\varphi_r$ | $\lvert \bigcap_d z_d \rvert \cdot \min_d \varphi_d / \lvert z_d \rvert$; summed analytically, never enumerated. |
| Φ-structure | $C(D)$ | Distinctions ∪ relations of a complex. |
| big phi | $\Phi$ | $\sum \varphi_d + \sum \varphi_r$ over the Φ-structure. |
| monad | — | A single-unit complex. |

## Macro terms (module `iitx.transforms`)

| Term | Definition |
|---|---|
| coarse-graining | Unit partition + per-macro-unit state grouping (+ temporal grain); macro TPM by fiber averaging $D^{-1} G^\top T^\tau G$. |
| grouping | The surjective, identity-irrelevant map from micro-states of a group to macro-states. |
| black-boxing | Box partition + designated output units (+ temporal grain); macro state = projection onto outputs at the window end; hidden units noised. |
| hidden / output units | Units of a box invisible to other boxes / read as the box's state. |
| temporal grain | $\tau$, the number of micro steps per macro step (`steps`). |
| supervenience | The macro state is a function of the micro state (holds for both transforms). |

## Naming rules (enforced by review and ruff)

1. **Cause/effect mirror**: any name containing `cause` has an `effect` twin; shared
   machinery takes `direction: Direction` instead of forking code paths.
2. **Qualified phis only**: `phi_c`, `phi_e`, `phi_s`, `phi_d`, `phi_r`, `big_phi` — a
   bare `phi` appears only as the `Measure.phi` method, whose docstring names which
   quantity it returns.
3. **`system` everywhere**: never `network`, `substrate`, or `subsystem` in public
   names; a candidate system is a `subset` mask over a `System`.
4. **Masks and indices**: `*_mask` is `(n,)` bool over units; `*_index` is an integer
   into a canonical enumeration table; state vectors are `state`, flat indices are
   `state_index`.
5. **`shape` means alphabets**: the per-unit alphabet tuple, chosen to coincide with the
   NumPy shape of full state-space arrays.
6. **Little-endian always**: any function touching state order says so in its docstring;
   big-endian exists only in explicitly named I/O converters, if ever.
7. **Paper symbols in docstrings**: every public docstring states the paper equation it
   implements (e.g. "Eq. 38, Albantakis et al. 2023") so a scientist can map code to
   theory in one hop.
