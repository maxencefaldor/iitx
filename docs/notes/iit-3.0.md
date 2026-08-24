# IIT 3.0 — Exhaustive Technical Notes

**Source:** Oizumi M, Albantakis L, Tononi G (2014). *From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0.* PLoS Comput Biol 10(5): e1003588. Including Text S1 (differences from earlier versions), Text S2 (Supplementary Methods — the canonical mathematical formulation), Text S3 (integrated vs. Shannon information), Figure S1 (exclusion at the mechanism level).

**Reference implementation named by the paper:** MATLAB program at `https://github.com/Albantakis/iit/tree/IIT-3.0-Program` (Text S2, ref. [7]). EMD computed with the fast EMD code of Pele & Werman (2008, 2009). Several details below are pinned down *only* by this implementation (and by its successor, PyPhi); every such gap is flagged in §10.

Notation used throughout these notes (matching the paper): superscripts $p$, $c$, $f$ denote *past* ($t_{-1}$), *current* ($t_0$), *future* ($t_{+1}$). $p^{\mathrm{per}}$ = "perturbed" (uniform / maximum-entropy) distribution imposed by intervention. $p^{\mathrm{uc}}$ = unconstrained repertoire. $[\,]$ = the empty set of elements. A "purview slash" expression $M^c/Z^p$ means: the cause repertoire of mechanism $M$ (in its current state) over purview $Z$ in the past; $M^c/Z^f$ is the effect analogue.

---

## 1. The system model

- **Discrete dynamical system.** A system $U$ of $N$ elements (the paper's examples: binary logic gates — OR, AND, XOR, MAJ, PARITY, COPY, NULL — and linear threshold units). Time is discrete, $t_{-1} \to t_0 \to t_{+1}$. Elements have (in all worked examples) binary states; the framework is stated to extend to non-binary states "as long as a distance between the individual states is given" (Text S2, EMD section).
- **TPM (transition probability matrix).** The mechanisms of the system determine a TPM specifying $p(s_t \mid s_{t-1})$ for every pair of system states. Crucially, the TPM is defined *interventionally*: it is "obtained by perturbing the system into all its possible states" (Box 1) — i.e. it is $p(s_t \mid do(s_{t-1}))$, not an observational stationary distribution. Deterministic systems give 0/1 entries; stochastic TPMs are explicitly allowed.
- **Conditional independence assumption** (Text S2, Eq. S1). Mechanisms are conditionally independent given the previous full state:

  $$p(ABC^t \mid ABC^{t-1}) \;=\; p(A^t \mid ABC^{t-1})\; p(B^t \mid ABC^{t-1})\; p(C^t \mid ABC^{t-1}).$$

  In general, for state $s^t = (s_1^t,\dots,s_n^t)$:

  $$p(s^t \mid s^{t-1}) \;=\; \prod_{i=1}^{n} p(s_i^t \mid s^{t-1}).$$

  Interpretation: no instantaneous interactions; causes precede effects. Mechanisms are 1st-order Markov functions of their inputs. This assumption is what makes all repertoires factorizable per element and lets virtual elements be dispensed with in practice (see §2).
  - *Implementation consequence:* the whole analysis can be driven from the $n$ per-element conditional distributions $p(s_i^t \mid s^{t-1})$ ("node TPMs"), each a tensor of shape $(2,\dots,2 \mid 2)$.
- **Current state.** The analysis is *state-dependent*: everything is computed for the system in its current state $s_0$ at $t_0$ (e.g. $ABC = 100$). The TPM column for $s_0$ gives possible past states; the row gives possible futures.
- **Candidate set and background conditions** (Text S2). A candidate set $S \subseteq U$ is the set under IIT analysis. Elements *outside* $S$ are **background conditions**: they are **not noised** — their connections stay intact and their states are **fixed at their actual values**: fixed at their past state ($t_{-1}$) when evaluating cause repertoires, and at their current state ($t_0$) when evaluating effect repertoires. The candidate set's effective TPM is the full TPM conditioned on these fixed background states; a different background state yields a different conditioned TPM and hence a possibly different conceptual structure, even with the identical internal state. (Text S1 point 9 stresses this is a change from earlier IIT versions, where outside elements were noised.)
  - Note (Text S2): in all the paper's examples, the state at $t_{-1}$ is taken equal to the current state at $t_0$; e.g. within $ABC$, when analyzing sub-candidate-sets it is assumed $ABC(t_{-1}) = 110$ etc. In general the background past and current states are two separate pieces of data.
- Elements inside the candidate set but outside a *mechanism* under consideration have **undetermined current state** for that mechanism (they are perturbed / treated as independent noise sources — see §2).

## 2. Cause and effect repertoires

Fix a candidate set $S$ (elements conditionally independent given the past, background conditions applied), a **mechanism** $M \subseteq S$ in its current state $m_0$, and a **purview** $Z \subseteq S$.

### 2.1 Cause repertoire

The **cause repertoire** $p_{\text{cause}}(z \mid m_0) \equiv p(Z^p = z \mid M^c = m_0)$ is the distribution over past purview states obtained by Bayes' rule under a **uniform interventional prior** on past states ("perturbing the set into all its states with equal likelihood", $p^{\mathrm{per}}$):

- Full purview ($Z = S$, e.g. $S = ABC$, $M = A$, $m_0 = 1$; Text S2 Eq. S2):

  $$p(ABC^p \mid A^c = 1) \;=\; \frac{p(A^c = 1 \mid ABC^p)\, p^{\mathrm{per}}(ABC^p)}{p(A^c = 1)},$$

  with $p^{\mathrm{per}}(ABC^p) = 1/8$ (uniform over the $2^n$ perturbed past states) and $p(A^c=1) = \sum_{abc} p(A^c=1 \mid ABC^p = abc)\, p^{\mathrm{per}}(abc)$ the normalizing constant.

- Sub-purview ($Z \subsetneq S$; Text S2 Eq. S3): one **marginalizes over the elements outside the purview, which remain unconstrained**:

  $$p(C^p \mid A^c = 1) \;=\; \frac{\Bigl(\sum_{AB^p} p(A^c = 1 \mid C^p, AB^p)\, p^{\mathrm{per}}(AB^p)\Bigr)\, p^{\mathrm{per}}(C^p)}{p(A^c = 1)} .$$

  I.e. the elements of $S \setminus Z$ at $t_{-1}$ are averaged out under the uniform perturbation distribution.

- **Virtual elements** (Text S2 Eq. S4, Fig. S2-1A). For a **higher-order mechanism** (e.g. $M = AB$, $m_0 = 10$) over a limited purview (e.g. $Z = AB^p$), naive marginalization over the excluded element $C^p$ can *induce spurious correlations* in $AB^c$ if $C$ sends common input to both $A$ and $B$. Since the aim is to assess what $M^c$ says about $Z^p$ *independent of* $C^p$, the marginalized element must be replaced by **virtual elements** $C^p_V$ — one independent copy of $C$ per output connection (a copy $C_A$ feeding $A$ and a copy $C_B$ feeding $B$), each independently perturbed uniformly:

  $$p(AB^p \mid AB^c = 10) \;=\; \frac{\Bigl(\sum_{C^p_V} p(AB^c = 10 \mid AB^p, C^p_V)\, p^{\mathrm{per}}(C^p_V)\Bigr)\, p^{\mathrm{per}}(AB^p)}{p(AB^c = 10)} .$$

  Virtual elements break common-input correlations: states 0 and 1 are imposed **independently over every output connection** of the excluded element.

- **Factorization shortcut** (Text S2 Eq. S5). Because mechanisms are conditionally independent 1st-order Markov functions, the cause repertoire of a higher-order mechanism equals the *product of the cause repertoires of its elementary sub-mechanisms*, which makes explicit virtual elements unnecessary in actual computations:

  $$p(AB^p \mid AB^c = 10) \;=\; p(AB^p \mid A^c = 1) \times p(AB^p \mid B^c = 0).$$

  ⚠ Both factors are normalized distributions over the *same* purview space $\Omega_{Z}$, so their pointwise product is **not automatically normalized**; the paper writes Eq. S5 without an explicit normalization constant. The reference implementation renormalizes the product (see §10, A2). Generic form for the library:

  $$p_{\text{cause}}(z \mid m_0) \;=\; \frac{1}{K} \prod_{i \in M} p_{\text{cause}}(z \mid m_{0,i}), \qquad K = \sum_{z \in \Omega_Z} \prod_{i \in M} p_{\text{cause}}(z \mid m_{0,i}).$$

- **Empty mechanism** ($M = [\,]$): the cause repertoire is the unconstrained cause repertoire (uniform) over $Z$ — this appears as parts like $[\,]/Z^p$ in partitions and as the null concept.

### 2.2 Effect repertoire

The **effect repertoire** $p_{\text{effect}}(z \mid m_0) \equiv p(Z^f = z \mid M^c = m_0)$ is the distribution over future purview states obtained by **fixing the mechanism at its current state** and **independently perturbing all other current-state elements uniformly** (virtual elements again used, per input connection, to avoid common-input correlations; Text S2 Eqs. S7–S9, Fig. S2-1B):

- Full purview:
  $$p(ABC^f \mid A^c = 1) \;=\; \sum_{BC^c_V} p(ABC^f \mid A^c = 1,\, BC^c_V)\; p^{\mathrm{per}}(BC^c_V).$$

- Factorization over *future* elements (exact under conditional independence; Text S2 Eq. S8):
  $$p(ABC^f \mid A^c = 1) \;=\; p(A^f \mid A^c = 1)\; p(B^f \mid A^c = 1)\; p(C^f \mid A^c = 1),$$
  where each single-future-element factor is (Eq. S9)
  $$p(A^f \mid A^c = 1) \;=\; \sum_{BC^c} p(A^f \mid A^c = 1,\, BC^c)\; p^{\mathrm{per}}(BC^c).$$

  Generic form: for each purview element $j \in Z$,
  $$p_{\text{effect}}(z \mid m_0) \;=\; \prod_{j \in Z} p(z_j^f \mid M^c = m_0), \qquad p(z_j^f \mid m_0) = \sum_{x \in \Omega_{S \setminus M}} p(z_j^f \mid m_0, x)\, p^{\mathrm{per}}(x).$$

  This product over *disjoint* future variables **is** automatically normalized (unlike the cause side).

- Elements of $S$ outside the future purview at $t_{+1}$ are simply dropped (marginalized trivially, since future elements are conditionally independent given $t_0$).

### 2.3 Unconstrained repertoires

- **Unconstrained cause repertoire** $p^{\mathrm{uc}}(Z^p)$: the cause repertoire in the absence of any mechanism — the **uniform distribution** over past purview states: $p^{\mathrm{uc}}(Z^p) = p^{\mathrm{per}}(Z^p) = 1/|\Omega_Z|$.
- **Unconstrained effect repertoire** $p^{\mathrm{uc}}(Z^f)$ (Text S2 Eqs. S11–S12): **not uniform.** It is the effect repertoire of the empty mechanism — the future distribution with *all inputs unconstrained*:

  $$p^{\mathrm{uc}}(ABC^f) \;=\; \sum_{ABC^c_V} p(ABC^f \mid ABC^c_V)\, p^{\mathrm{per}}(ABC^c_V) \;=\; \prod_{j} \Bigl(\sum_{ABC^c} p(j^f \mid ABC^c)\, p^{\mathrm{per}}(ABC^c)\Bigr).$$

  E.g. for an OR gate $A$ with two inputs perturbed uniformly over $[00,01,10,11]$: $p^{\mathrm{uc}}(A^f{=}1) = 0.75$.

### 2.4 Cause/effect information (mechanism-level information postulate)

With $D$ = EMD (see §3.3):

$$ci(Z^p \mid m_0) = D\bigl(p(Z^p \mid m_0)\,\|\, p^{\mathrm{uc}}(Z^p)\bigr), \qquad
ei(Z^f \mid m_0) = D\bigl(p(Z^f \mid m_0)\,\|\, p^{\mathrm{uc}}(Z^f)\bigr),$$

$$cei(Z^{p,f} \mid m_0) \;=\; \min\bigl[ci(Z^p \mid m_0),\; ei(Z^f \mid m_0)\bigr] \qquad \text{(main text Eqs. 1–3)}.$$

The $\min$ implements the "intrinsic information bottleneck": an element with only causes (no outputs into $S$) or only effects (no inputs from $S$) has $cei = 0$. (Worked values: $ci(ABC^p|A^c{=}1) = 0.33$, $ei(ABC^f|A^c{=}1) = 0.25$, $cei = 0.25$.) $cei$ is conceptually motivating but is **not** used downstream — $\varphi$ (below) is what everything is built from.

## 3. Mechanism partitions, small phi ($\varphi$), and MICE

### 3.1 Partitions of a mechanism–purview pair

A **partition** severs the mechanism–purview dependency by cutting the pair $(M, Z)$ into two parts and rendering cross-part connections causally inert: "elements outside of the part under consideration become unconstrained and thus effectively act as independent noise sources — they are *injected with noise*" (Text S2 "Partitions"). Form (as used in all the paper's examples):

$$\frac{M^c}{Z^{p/f}} \;\to\; \frac{M_1^c}{Z_1^{p/f}} \times \frac{M_2^c}{Z_2^{p/f}}, \qquad M_1 \uplus M_2 = M,\quad Z_1 \uplus Z_2 = Z,$$

with parts allowed to have an **empty mechanism** ($[\,]/Z_i$, meaning the unconstrained repertoire of $Z_i$) — the paper's future MIP example is $ABC^c/ABC^f \to (ABC^c/AC^f) \times ([\,]/B^f)$ — and, in the paper's examples, both purview parts nonempty. The partitioned repertoire is the **product of the parts' repertoires**, each computed exactly as in §2 (Text S2 Eqs. S13–S15):

$$p(Z^{p/f} \mid m_0 / P) \;=\; p\bigl(Z_1^{p/f} \mid m_{0,1}\bigr) \times p\bigl(Z_2^{p/f} \mid m_{0,2}\bigr).$$

Worked example (main text Eqs. 4–5, mechanism $ABC = 100$):
$$p(ABC^p \mid ABC^c{=}100/\mathrm{MIP}) = p(C^p \mid AB^c{=}10) \times p(AB^p \mid C^c{=}0),$$
$$p(ABC^f \mid ABC^c{=}100/\mathrm{MIP}) = p(AC^f \mid ABC^c{=}100) \times p(B^f).$$

⚠ The paper never enumerates the full set of admissible partitions (only bipartitions? are parts with nonempty mechanism and empty purview allowed? is the "total partition" $M/[\,] \times [\,]/Z$ included?). This is fixed only by the reference implementation — see §10, A3.

### 3.2 MIP and $\varphi$

The **minimum information partition (MIP)** of a mechanism–purview pair is the partition making the *least* difference to the repertoire ("minimum difference partition"), **evaluated without normalization** (Text S1 point 5 — a deliberate change from IIT 1.0/2.0):

$$\mathrm{MIP}_{\text{cause}}(m_0, Z) = \arg\min_{P}\; D\bigl(p(Z^p \mid m_0)\,\|\,p(Z^p \mid m_0/P)\bigr),$$

and analogously for the effect side (the cause MIP and effect MIP are found **separately** and generally differ). Then

$$\varphi^{\mathrm{MIP}}_{\text{cause}}(Z^p \mid m_0) = D\bigl(p(Z^p \mid m_0)\,\|\,p(Z^p \mid m_0/\mathrm{MIP})\bigr), \qquad
\varphi^{\mathrm{MIP}}_{\text{effect}}(Z^f \mid m_0) = D\bigl(p(Z^f \mid m_0)\,\|\,p(Z^f \mid m_0/\mathrm{MIP})\bigr),$$

$$\varphi^{\mathrm{MIP}}(Z^{p,f} \mid m_0) \;=\; \min\bigl[\varphi^{\mathrm{MIP}}_{\text{cause}},\; \varphi^{\mathrm{MIP}}_{\text{effect}}\bigr] \qquad \text{(main text Eqs. 6–8)}.$$

(The superscript MIP is dropped thereafter: $\varphi$ always means $\varphi^{\mathrm{MIP}}$.) Worked values for $ABC{=}100$ over the full purview: $\varphi_{\text{cause}} = 0.5$, $\varphi_{\text{effect}} = 0.25$, $\varphi = 0.25$.

### 3.3 The distance $D$: earth mover's distance (EMD)

- $D$ = EMD = Wasserstein-1 distance between the two repertoires (replacing the KLD of IIT 2.0; motivation in Text S2: KLD is asymmetric, unbounded, and insensitive to *which* states differ).
- **Ground distance:** the **Hamming distance between purview/system states** (number of elements whose state differs; e.g. $d(000,111) = 3$, $d(010,100) = 2$).
- Bounded by the number of elements $N$ (for binary elements); symmetric; a true metric.
- "In principle extendable to non-binary states, as long as a distance between the individual states is given, which is an intrinsic property of the mechanisms under consideration" — the non-binary ground metric is *not* specified.
- Computation: min-cost transportation LP; the paper used Pele–Werman's fast EMD MATLAB code.

### 3.4 Purview search: core cause, core effect, MICE, concept ($\varphi^{\mathrm{Max}}$)

Mechanism-level **exclusion**: a mechanism can have only *one* cause and *one* effect — those that are maximally irreducible; no causal superposition.

- Evaluate $\varphi_{\text{cause}}(Z^p \mid m_0)$ for **every purview in the power set** of past purviews $\mathcal{P} = \{A^p, B^p, C^p, AB^p, AC^p, BC^p, ABC^p, \dots\}$ (nonempty subsets of the candidate set). The purview achieving the maximum is the **core cause**; its repertoire is the **maximally irreducible cause repertoire**, with value $\varphi^{\mathrm{Max}}_{\text{cause}}(\mathcal{P} \mid m_0) = \max_{Z \in \mathcal{P}} \varphi^{\mathrm{MIP}}_{\text{cause}}(Z^p \mid m_0)$.
- Same over future purviews $\mathcal{F}$ → **core effect**, $\varphi^{\mathrm{Max}}_{\text{effect}}(\mathcal{F} \mid m_0)$.
- $$\varphi^{\mathrm{Max}}(m_0) \;=\; \min\bigl[\varphi^{\mathrm{Max}}_{\text{cause}}(\mathcal{P} \mid m_0),\; \varphi^{\mathrm{Max}}_{\text{effect}}(\mathcal{F} \mid m_0)\bigr] \qquad \text{(main text Eq. 9)}.$$
  (Note the order of operations: max over purviews on each side *first*, then min across sides.)
- The pair (maximally irreducible cause repertoire, maximally irreducible effect repertoire) is the **MICE**. A mechanism with $\varphi^{\mathrm{Max}} > 0$ **constitutes a concept** (a.k.a. *core concept*, quale *sensu stricto*). If $\varphi^{\mathrm{Max}} = 0$ the concept "simply does not exist" (e.g. mechanism $AC$ in the worked $ABC$ example).
- Worked example: concept of $A{=}1$ has core cause $A^c/BC^p$, core effect $A^c/B^f$, $\varphi^{\mathrm{Max}} = 0.17$; concept of $BC{=}00$ has core cause $BC^c/AB^p$ with $\varphi^{\mathrm{Max}}_{\text{cause}} = 0.33$.
- Fig. S1 motivation: a neuron with strong synapses $S_1, S_2$ and many weak synapses — the core cause is "the strong synapses", cutting the infinite regress of ever-larger candidate causes (causal Occam's razor).
- Exclusion applies to purviews *of a single mechanism*, not across mechanisms: elementary and higher-order mechanisms all keep their own concepts (composition postulate).
- ⚠ Ties (two purviews with equal $\varphi^{\mathrm{Max}}$) are never discussed — see §10, A5.

## 4. Concepts, conceptual structure, and big Phi ($\Phi$)

### 4.1 Conceptual structure (constellation) and concept space

- The **conceptual structure** $C$ (constellation) of a candidate set $S$ in state $s_0$ is the set of *all* its concepts: every mechanism $M \in 2^S \setminus \{\emptyset\}$ with $\varphi^{\mathrm{Max}}(m_0) > 0$, each carrying its MICE and its $\varphi^{\mathrm{Max}}$ value.
- **Concept space**: a $2 \cdot |\Omega_S|$-dimensional space (one axis per possible past state and per possible future state of $S$; for $n=3$ binary elements: $16$ axes). A concept is a point whose coordinates are the probabilities its cause-effect repertoire assigns to each past/future state, with "size" $\varphi^{\mathrm{Max}}$.
  - ⚠ Implicit requirement: to place a concept (whose purviews are generally proper subsets of $S$) in this space, its cause and effect repertoires must be **expanded over the whole candidate set** — non-purview elements filled in with the corresponding unconstrained distribution (uniform on the cause side, unconstrained-inputs product on the effect side). The paper never states this rule; the reference implementation defines it. See §10, A4.
- The **null concept**: the unconstrained cause-effect repertoire $p^{\mathrm{uc}} = (p^{\mathrm{uc}}(S^p),\, p^{\mathrm{uc}}(S^f))$ of the candidate set, "a concept that specifies nothing", with $\varphi = 0$.

### 4.2 Conceptual information (CI)

$$CI(C \mid s_0) \;=\; D\bigl(C \,\|\, p^{\mathrm{uc}}(S^{p,f})\bigr) \;=\; \sum_{\text{concepts } c \in C} \varphi^{\mathrm{Max}}(c)\; d\bigl(c,\, p^{\mathrm{uc}}\bigr) \qquad \text{(main text Eq. 10)},$$

where $d(c_1, c_2)$, the **distance between two concepts in concept space**, is the sum of two standard EMDs:

$$d(c_1, c_2) \;=\; \mathrm{EMD}\bigl(p_{\text{cause}}(c_1),\, p_{\text{cause}}(c_2)\bigr) \;+\; \mathrm{EMD}\bigl(p_{\text{effect}}(c_1),\, p_{\text{effect}}(c_2)\bigr),$$

both over full-set-expanded repertoires with Hamming ground distance. (Worked value: $CI(C \mid ABC{=}100) = 2.11$.)

### 4.3 System-level integration: unidirectional cuts and $\Phi$

- **System partition (cut):** noising the connections **from** one subset $S_1 \subseteq S$ **to** its complement $S \setminus S_1$ — a **unidirectional** cut ($S_1 \to S \setminus S_1$; the outgoing connections of $S_1$ are replaced by independent noise / virtualized-unconstrained inputs into $S \setminus S_1$). For each subset both directions must be evaluated ($S_1 \to S\setminus S_1$ and $S\setminus S_1 \to S_1$); the MIP is the minimum over all of these. Rationale for unidirectionality: it detects "weakly integrated" appendices — a subset with only causes or only effects in the rest is not an integral part (board-of-directors analogy; Fig. 13). A set is (strongly) integrated iff **every subset has both causes and effects in its complement**; equivalently, if any unidirectional cut costs nothing, $\Phi = 0$.
- The **partitioned constellation** $C^{\mathrm{MIP}}_{\rightarrow}$: the *entire* conceptual structure recomputed for the cut system (all mechanisms, purviews, mechanism-MIPs re-evaluated under the cut TPM). Text S1 point 4 stresses this whole-constellation assessment is the key change from IIT 1.0/2.0 (which partitioned only the highest-order concept). Through the cut, concepts may change location, lose $\varphi^{\mathrm{Max}}$, or disappear.
- **Integrated conceptual information** (main text Eq. 11):

  $$\Phi^{\mathrm{MIP}}(C \mid s_0) \;=\; D\bigl(C \,\big\|\, C^{\mathrm{MIP}}_{\rightarrow}\bigr), \qquad
  \mathrm{MIP} = \arg\min_{\text{unidirectional cuts}} D\bigl(C \,\|\, C_{\rightarrow}\bigr),$$

  with the superscript MIP subsequently dropped: $\Phi \equiv \Phi^{\mathrm{MIP}}$.

### 4.4 Extended EMD between constellations (Text S2, "Distance for constellations of concepts")

The distance $D(C \,\|\, C^{\mathrm{MIP}}_{\rightarrow})$ is an **extended EMD** — a transportation problem *in concept space*:

- The "earth" is $\varphi^{\mathrm{Max}}$: supplies are the $\varphi^{\mathrm{Max}}$ values of the concepts of $C$; demands are the $\varphi^{\mathrm{Max}}$ values of the concepts of $C^{\mathrm{MIP}}_{\rightarrow}$.
- The ground distance between a concept of $C$ and a concept of $C^{\mathrm{MIP}}_{\rightarrow}$ is $d(c_1, c_2)$ from §4.2 (sum of cause-side and effect-side EMDs of their repertoires).
- **Disappearing concepts / unbalanced mass:** since $\sum \varphi^{\mathrm{Max}}(C)$ is usually larger than $\sum \varphi^{\mathrm{Max}}(C^{\mathrm{MIP}}_{\rightarrow})$, **any residual $\varphi^{\mathrm{Max}}$ is assigned to the "null" concept $p^{\mathrm{uc}}$, which is included as an additional location in $C^{\mathrm{MIP}}_{\rightarrow}$** (with unlimited capacity). So a concept destroyed by the partition contributes $\varphi^{\mathrm{Max}}(c) \cdot d(c, p^{\mathrm{uc}})$, where $d(c, p^{\mathrm{uc}})$ = EMD(cause rep., $p^{\mathrm{uc},p}$) + EMD(effect rep., $p^{\mathrm{uc},f}$).
- Unaffected concepts (identical before and after the cut) contribute 0.
- "In the general case, the optimal way to redistribute $\varphi^{\mathrm{Max}}$ from $C$ to $C^{\mathrm{MIP}}_{\rightarrow}$ must be found using an optimization algorithm" — i.e. a genuine (unbalanced, via the null sink) min-cost transport LP over up to $2^n - 1$ supply points and $2^n$ demand points.
- Worked example (Text S2 Fig. S2-4, candidate set $ABC$, cut $AB \to C$): concepts $A$, $B$ survive unchanged (distance 0); concepts $C$, $AB$, $BC$, $ABC$ are destroyed, their $\varphi^{\mathrm{Max}}$ transported to $p^{\mathrm{uc}}$:
  $\Phi^{\mathrm{Max}} = 0.5{\cdot}2 + 0.33{\cdot}1.25 + 0.25{\cdot}1 + 0.25{\cdot}1 = 1.92$.
- ⚠ Underspecified corners: what if the partitioned constellation carries *more* total $\varphi$ than the whole ("usually higher" is not "always"); whether *new* concepts appearing only in the cut system are possible and how they are handled. See §10, A6–A7.

## 5. Exclusion at the system level: complexes

- To find complexes, $\Phi$ must be evaluated **for every candidate set** — all subsets of the system (Fig. 14 evaluates all subsets of $ABC$ plus supersets such as $ABCD$), each analysis treating the elements outside the candidate as fixed background conditions.
- **Single elements are excluded as candidate sets** ("they cannot be partitioned and thus cannot be complexes by definition") → candidates are subsets of size $\ge 2$.
- A **complex** is a candidate set that is a **local maximum of $\Phi$**: its $\Phi^{\mathrm{Max}}$ is maximal "as compared to all overlapping sets of elements" (supersets, subsets, and partially overlapping sets).
- **Exclusion postulate:** complexes cannot overlap; each element belongs to at most one complex at a time. A strongly integrated set (e.g. $ABCDE$ with $\Phi > 0$) is *excluded* from being a complex if it overlaps a set with higher $\Phi^{\mathrm{Max}}$ (e.g. $ABC$).
- **Major complex** (a.k.a. main complex): the complex with the highest $\Phi^{\mathrm{Max}}$ overall. **Minor complexes:** other, non-overlapping local maxima (e.g. system condenses into major complex $ABC$, minor complexes $DE$ and $FG$, plus residual interactions and purely feed-forward "unconscious" chains).
- Once a set is a complex, its concept space is **qualia space** and its MICS (constellation) is its quale *sensu lato*; $\Phi^{\mathrm{Max}}$ is the quantity/level of the experience, the constellation's shape its quality.
- Maxima should in principle be taken not only over element subsets but also over **spatio-temporal grains** (ref. [20]); the paper assumes the given binary elements and time step are optimal.
- ⚠ **Ties:** the paper never specifies what happens when overlapping candidate sets have exactly equal maximal $\Phi$ (nor equal-$\Phi$ system cuts). See §10, A5.

## 6. Glossary of symbols and quantities

| Symbol / term | Paper's name | Definition (pointer) |
|---|---|---|
| TPM | transition probability matrix | $p(s_t \mid do(s_{t-1}))$, via exhaustive perturbation (§1) |
| $s_0$ | current state | state of candidate set at $t_0$ |
| candidate set | — | subset $S \subseteq U$ under analysis; rest = background conditions |
| background conditions | — | outside elements fixed at actual past ($t_{-1}$) / current ($t_0$) states, not noised |
| mechanism $M$ | — | any nonempty subset of $S$ with a causal role, in state $m_0$ |
| purview $Z$ | — | subset of $S$ over which a repertoire is computed |
| $p^{\mathrm{per}}$ | perturbed distribution | uniform interventional distribution over states |
| $p(Z^p \mid m_0)$ | cause repertoire | §2.1 |
| $p(Z^f \mid m_0)$ | effect repertoire | §2.2 |
| $p^{\mathrm{uc}}(Z^p)$ | unconstrained past repertoire | uniform (§2.3) |
| $p^{\mathrm{uc}}(Z^f)$ | unconstrained future repertoire | unconstrained-inputs product, ≠ uniform (§2.3) |
| virtual elements ($C_V$) | — | per-connection independent noised copies of excluded elements (§2.1) |
| $ci$, $ei$ | cause / effect information | EMD to unconstrained repertoire (§2.4) |
| $cei$ | cause-effect information | $\min(ci, ei)$ |
| $D$ | distance | EMD (Wasserstein), Hamming ground distance (§3.3); extended EMD between constellations (§4.4) |
| MIP | minimum information partition | least-difference partition (mechanism level §3.2; system level §4.3), no normalization |
| $\varphi^{\mathrm{MIP}}_{\text{cause}}$, $\varphi^{\mathrm{MIP}}_{\text{effect}}$ | integrated cause / effect information | EMD(whole ‖ MIP-partitioned repertoire) |
| $\varphi$ ("small phi") | integrated information (mechanism) | $\min(\varphi_{\text{cause}}, \varphi_{\text{effect}})$ at the MIP |
| core cause / core effect | — | purview maximizing $\varphi_{\text{cause}}$ / $\varphi_{\text{effect}}$ over the power set |
| MICE | maximally irreducible cause-effect repertoire | core cause + core effect repertoires (quale *sensu stricto*) |
| $\varphi^{\mathrm{Max}}$ | — | $\min(\varphi^{\mathrm{Max}}_{\text{cause}}, \varphi^{\mathrm{Max}}_{\text{effect}})$; concept exists iff $> 0$ |
| concept (core concept) | — | mechanism + its MICE + $\varphi^{\mathrm{Max}}$ |
| $C$ | conceptual structure / constellation | set of all concepts of $S$ in $s_0$ |
| concept space / qualia space | — | $2\,|\Omega_S|$-dimensional; axes = past & future states |
| null concept | — | $p^{\mathrm{uc}}(S^{p,f})$, $\varphi = 0$ |
| $CI$ | conceptual information | $\sum_c \varphi^{\mathrm{Max}}(c)\, d(c, p^{\mathrm{uc}})$ |
| $C_{\rightarrow}$, $C^{\mathrm{MIP}}_{\rightarrow}$ | partitioned constellation | constellation of the unidirectionally cut set |
| $\Phi$, $\Phi^{\mathrm{MIP}}$ ("big phi") | integrated conceptual information | extended EMD $D(C \| C^{\mathrm{MIP}}_{\rightarrow})$ at the system MIP |
| $\Phi^{\mathrm{Max}}$ | — | $\Phi$ of a local-maximum candidate set |
| complex / major (main) complex / minor complex | — | local maxima of $\Phi$; the global one is major |
| MICS | maximally irreducible conceptual structure | constellation of a complex = quale *sensu lato* |
| strong vs. weak integration | — | every subset has causes **and** effects in the rest vs. one direction only |

## 7. Computational structure: the combinatorial axes

For a universe of $N$ binary elements and a candidate set of $n$ elements:

1. **System states**: the analysis is per current state $s_0$; there are $2^N$ states of the universe (and $2^n$ of a candidate set). The TPM itself is $2^N \times 2^N$ (or $N$ node-TPMs of size $2^N \times 2$). Background conditions add a dependence on the (past, current) outside states.
2. **Candidate systems**: all subsets of size $\ge 2$: $\;2^N - N - 1$. Each has its own conditioned TPM.
3. **Mechanisms** per candidate set: all nonempty subsets: $2^n - 1$.
4. **Purviews** per mechanism: nonempty subsets, independently for cause and effect side: $2^n - 1$ each ⇒ $(2^n - 1)^2$ mechanism–purview pairs per side across the candidate set (each requiring a repertoire of size $2^{|Z|}$, expanded to $2^n$ for concept-space comparisons).
5. **Mechanism partitions** per $(M, Z)$ pair: bipartitions $\{(M_1,Z_1),(M_2,Z_2)\}$ with $M_1 \uplus M_2 = M$, $Z_1 \uplus Z_2 = Z$; ordered assignments $2^{|M|} \cdot 2^{|Z|}$, i.e. on the order of $2^{|M|+|Z|-1}$ unordered candidates before excluding trivial/disallowed ones (exact admissible set: implementation-defined, §10 A3). Summed over all $(M,Z)$ pairs this is the dominant $O(5^n)$-ish inner loop.
6. **System cuts** per candidate set: one unidirectional cut per nonempty proper subset ($S_1 \to S \setminus S_1$): $2^n - 2$ cuts; **each cut requires recomputing the entire constellation** (axes 3–5) under the cut TPM.
7. **EMD instances**: every $\varphi$ evaluation is an LP over distributions of size up to $2^n$ with a $2^n \times 2^n$ Hamming cost matrix; every $\Phi$ evaluation is a transport LP over up to $(2^n - 1) + 1$ concept locations whose cost matrix entries are themselves pairs of $2^n$-dim EMDs.

Rough total: per state, per candidate set, the work is $\sum_{M,Z} (\text{partitions}) \times (\text{EMD})$, then $\times (2^n - 2)$ cuts, then $\times$ candidates $\times$ states. Everything is exponential stacked on exponential; exhaustive IIT 3.0 is feasible only for small $n$ (the paper's examples: $n \le 10$ with most analysis at $n \le 6$).

## 8. What the paper's examples pin down numerically (useful as golden tests)

- Candidate set $ABC$ (A = OR, B = AND, C = XOR; state $100$; background $DEF = 010$ at $t_{-1}$ and $t_0$):
  - $ci(ABC^p \mid A^c{=}1) = 0.33$; $ei(ABC^f \mid A^c{=}1) = 0.25$; $cei = 0.25$.
  - Mechanism $ABC{=}100$: cause MIP $(AB^c/C^p) \times (C^c/AB^p)$, $\varphi_{\text{cause}} = 0.5$; effect MIP $(ABC^c/AC^f) \times ([\,]/B^f)$, $\varphi_{\text{effect}} = 0.25$; $\varphi = 0.25$.
  - Concept $A{=}1$: core cause $A^c/BC^p$, core effect $A^c/B^f$, $\varphi^{\mathrm{Max}} = 0.17$.
  - Concept $BC{=}00$: core cause $BC^c/AB^p$, $\varphi^{\mathrm{Max}}_{\text{cause}} = 0.33$.
  - Mechanism $AC$: $\varphi^{\mathrm{Max}} = 0$ (no concept). Six concepts total: $A, B, C, AB, BC, ABC$ with $\varphi^{\mathrm{Max}} = 0.17, 0.17, 0.25, 0.25, 0.33, 0.5$.
  - $CI(C \mid 100) = 2.11$; system MIP = cut $AB \to C$; $\Phi^{\mathrm{Max}} = 0.5{\cdot}2 + 0.33{\cdot}1.25 + 0.25{\cdot}1 + 0.25{\cdot}1 = 1.92$; concept-to-null distances: $d(ABC, p^{\mathrm{uc}}) = 2$, $d(BC, p^{\mathrm{uc}}) = 1.25$, $d(AB, \cdot) = d(C, \cdot) = 1$.
  - $ABC$ is the (major) complex among all candidate subsets/supersets.
- EMD toy example (Text S2 Fig. S2-3): intact cause repertoire concentrated on $00$; Partition 1 spreads to $\{00, 10\}$, Partition 2 to $\{00, 11\}$; KLD = 1 bit for both, EMD = 0.5 vs. 1.
- Unconstrained effect repertoire of OR gate $A$ (2 inputs): $p(A{=}0) = 0.25,\ p(A{=}1) = 0.75$; AND gate $B$: $0.75/0.25$; XOR $C$: $0.5/0.5$.

## 9. Differences from earlier IIT versions worth remembering (Text S1)

1. Both causes *and* effects required (IIT 2.0 was cause-only); $\min$ over the two sides everywhere.
2. Elements of concepts are *mechanisms in a state*, not connections.
3. Irreducibility of a set assessed on the **whole conceptual structure** (all concepts of the power set), not just the highest-order concept.
4. **MIP evaluated without normalization.**
5. Concepts must themselves be irreducible ($\varphi^{\mathrm{Max}} > 0$).
6. KLD → EMD (proper metric on concept space), plus the extended constellation EMD.
7. Exclusion applied at the mechanism level too (MICE).
8. Background elements **fixed at actual values**, not noised.

## 10. Ambiguities / underspecified points (implementation-defined freedom)

**A1 — TPM provenance.** The TPM is defined by exhaustive perturbation. For a candidate set, the conditioned TPM depends on the background's past *and* current states; the paper's examples silently set past state = current state. A library must accept $s_{t-1}^{\text{bg}}$ and $s_{t_0}^{\text{bg}}$ (or a convention) explicitly.

**A2 — Normalization of the cause-repertoire product.** Eq. S5 (product of elementary cause repertoires over a shared purview) and Eq. 4 / S13-style partitioned cause products are written without normalization constants, yet must be normalized to be distributions (and for EMD to be balanced). The reference implementation renormalizes; a library must do so and document it. (Effect-side products over disjoint future elements need no renormalization. Edge case: if the normalization constant is 0 — mechanism state impossible under the perturbation — the repertoire is undefined; reference implementations fall back to conventions.)

**A3 — The set of mechanism partitions enumerated.** The paper says "of the many possible ways to partition a mechanism" and shows only *bipartitions* of $(M, Z)$ with possibly-empty mechanism parts ($[\,]/Z_i$). It never states: (i) whether only bipartitions are considered (yes, in the reference implementation) vs. finer partitions; (ii) whether a part may have a *nonempty mechanism and empty purview* ($M_i/[\,]$); (iii) whether the "total partition" $(M/[\,]) \times ([\,]/Z)$ is included (this determines whether $\varphi$ can exceed $cei$-like quantities); (iv) uniqueness/ordering conventions. All of this is fixed only by the MATLAB program / PyPhi. Decide and document; expose as options if fidelity to multiple conventions is desired.

**A4 — Expansion of repertoires to the full space.** Concept-space coordinates, $d(c_1, c_2)$, $CI$, and the extended EMD all require comparing repertoires computed over *different* purviews. The necessary rule — expand each repertoire over the whole candidate set by filling non-purview elements with the appropriate unconstrained distribution (uniform for past, unconstrained-inputs product for future) — is nowhere stated in the paper; it is defined by the reference implementation.

**A5 — Tie-breaking, everywhere.** The paper is silent on ties in: (i) the mechanism MIP (two partitions with equal minimal EMD — matters only if the resulting partitioned repertoires differ, which they can); (ii) the MICE purview choice (two purviews with equal $\varphi^{\mathrm{Max}}$ — the chosen purview changes the concept's repertoire, hence changes $CI$ and $\Phi$!); (iii) the system MIP; (iv) equal $\Phi^{\mathrm{Max}}$ among overlapping candidate sets (which one is *the* complex?). Reference implementations resolve these by enumeration order or by secondary criteria (e.g. PyPhi's later conventions: prefer larger/smaller purviews); IIT 3.0 itself gives no rule. These ties are not measure-zero: symmetric networks hit them constantly. A JAX implementation needs an explicit, deterministic, documented tie-breaking policy.

**A6 — Extended EMD when partitioned $\varphi$ mass exceeds the whole's.** Text S2 says $\sum \varphi^{\mathrm{Max}}(C)$ is "usually higher" than that of $C^{\mathrm{MIP}}_{\rightarrow}$ and routes the *residual* to the null concept. The converse case (cut system with more total $\varphi$, e.g. new or strengthened concepts after a cut) is not addressed — is the null concept also a supply node? Implementation-defined.

**A7 — New concepts under a cut.** Concepts "may change location, lose $\varphi^{\mathrm{Max}}$, or disappear"; the possibility of concepts *appearing* only in the partitioned constellation is not discussed, but nothing formally rules it out. How they enter the transport problem is implementation-defined.

**A8 — EMD details.** (i) Ground distance is Hamming for binary states; for non-binary elements the paper defers to "an intrinsic property of the mechanisms" without definition. (ii) Exact vs. approximate EMD solvers, numerical precision, and the resulting equality thresholds (which interact with tie-breaking and with the $\varphi > 0$ concept-existence test) are unspecified. Define an explicit tolerance $\varepsilon$ for "equal" and "zero".

**A9 — Empty purviews.** Whether $\varphi_{\text{cause}}$/$\varphi_{\text{effect}}$ maximization ranges over the empty purview (and what a repertoire over the empty purview means) is unstated; reference implementations exclude empty purviews from the MICE search (while empty *mechanism* parts do occur inside partitions).

**A10 — Candidate-set constraints.** Single elements are excluded as complexes by fiat. Whether candidate sets that are not weakly connected should be pre-filtered (they trivially give $\Phi = 0$) is an optimization, not a definition. Also, "local maximum ... over elements, space, and time": the spatio-temporal-grain search (macro/micro) is acknowledged but entirely out of scope of the given formalism.

**A11 — State-dependence bookkeeping.** $\Phi^{\mathrm{Max}}$ is a property of a (set, state) pair. The paper does not define any state-averaged quantity in IIT 3.0 (unlike IIT 2.0); comparisons across states are left to the user.

**A12 — cei vs. φ.** $cei$ (min of $ci$, $ei$) and $CI$ are defined and motivated but play no role in the $\Phi$ algorithm; they are diagnostics. Do not confuse $CI$ (distance of constellation to null concept) with $\Phi$ (distance of constellation to its MIP constellation).
