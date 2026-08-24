# IIT 4.0 — Technical Notes for `iitx`

Exhaustive mathematical notes on the IIT 4.0 formalism and the intrinsic difference (ID)
measure, compiled as the foundation for the `iitx` JAX library.

**Primary sources** (read in full, including supplementary material):

1. Albantakis L, Barbosa L, Findlay G, Grasso M, Haun AM, Marshall W, Mayner WGP,
   Zaeemzadeh A, Boly M, Juel BE, Sasai S, Fujii K, David I, Hendren J, Lang JP, Tononi G
   (2023). *Integrated information theory (IIT) 4.0: Formulating the properties of
   phenomenal existence in physical terms.* PLoS Comput Biol 19(10): e1011465.
   Including: S1 Text (tie resolution), S2 Text (comparison to IIT 1.0–3.0), S3 Text
   (analytical results for relations), S1 Fig (algorithm), S1 Notes (footnotes).
2. Barbosa LS, Marshall W, Streipert S, Albantakis L, Tononi G (2020). *A measure for
   intrinsic information.* Sci Rep 10, 18803. Including the Supplementary Material
   (formal theorem statement and uniqueness proof).

Secondary sources referenced by the 4.0 paper for pieces of the formalism (cited here
where the paper defers to them):

- Marshall W, Grasso M, Mayner WGP, Zaeemzadeh A, Barbosa LS, Chastain E, et al. (2023).
  *System Integrated Information.* Entropy 25. — origin of the φ_s formalism and the
  system-partition normalization proof.
- Barbosa LS, Marshall W, Albantakis L, Tononi G (2021). *Mechanism Integrated
  Information.* Entropy 23(3):362. — origin of the mechanism-level (distinction)
  formalism and the disintegrating partition set.
- Haun AM, Tononi G (2019). *Why Does Space Feel the Way it Does?* Entropy 21(12):1160.
  — first explicit account of relations (superseded in detail by 4.0).
- Albantakis L, Marshall W, Hoel E, Tononi G (2019). *What caused what?* Entropy
  21(5):459. — causal marginalization / product-probability machinery.
- Mayner WGP et al. (2018). *PyPhi: A toolbox for integrated information theory.* PLoS
  Comput Biol 14(7):e1006343. — reference implementation; IIT 4.0 examples computed with
  the `iit-4.0` feature branch of PyPhi (now released as PyPhi's IIT 4.0 support).

Notation: logs are **base 2 throughout**. Integrated/intrinsic information is measured
in **ibits** ("intrinsic bits": a point-wise bit value weighted by a probability).
`|·|₊` denotes the positive-part operator, `max(·, 0)`. Uppercase letters ($S$, $M$,
$Z$…) denote sets of units (random variables); lowercase ($s$, $m$, $z$…) denote those
sets *in a state* — formally sets of (unit, state) tuples, so set operations
(⊆, ∩, ∪, \) act on unit–state pairs and are state-sensitive. Bars denote the state at
the other time step ($\bar s$ = a potential cause state at $t-1$ or effect state at
$t+1$ of current state $s$ at $t$).

---

## 1. The Intrinsic Difference (ID) measure (Barbosa et al. 2020)

### 1.1 Setup

Let $X$ (symbol selected by a source) and $Y$ (symbol received) be discrete random
variables with common support $\Omega$, $|\Omega| = n$. Given an observed symbol
$y_\beta$, define two distributions over the $n$ possible source symbols:

$$
P^n = [p_1, \ldots, p_n], \qquad p_\alpha = P(X = x_\alpha \mid Y = y_\beta),
$$
$$
Q^n = [q_1, \ldots, q_n], \qquad q_\alpha = P(X = x_\alpha).
$$

$Q^n$ is the "chance" distribution — what would obtain if there were **no causal
connection** between source and receiver. Special distributions:
$V^n = [1, 0, \ldots, 0]$ (noiseless) and $U^n = [1/n, \ldots, 1/n]$ (uniform).
A channel is *noiseless* if $P^n = V^n \ne Q^n$ and *fully noisy* if
$P^n = U^n = Q^n$. The Kronecker product of channels is written
$P^l * Q^m = (p_1 q_1, \ldots, p_1 q_m, \ldots, p_l q_1, \ldots, p_l q_m)$.

### 1.2 The three defining properties (informal → formal)

**Causality** — the measure is zero iff the source has no causal connection to the
received symbol:
$$
D(P^n, Q^n) = 0 \iff P^n \equiv Q^n .
$$

**Specificity** — the measure must reflect the information carried by **one specific
symbol** (state), not an average over symbols. With
$f(p_\alpha, q_\alpha)$ the per-state information function:
$$
D(P^n, Q^n) = \max_{\alpha \in \{1,\ldots,n\}} f(p_\alpha, q_\alpha).
$$

**Intrinsicality** — extension of a channel by an independent channel must behave as:

- *Expansion* (additivity for noiseless extension): if a noiseless channel is extended
  by an independent noiseless channel,
  $$
  D(V_1^n * V_2^m,\; Q_1^n * Q_2^m) = D(V_1^n, Q_1^n) + D(V_2^m, Q_2^m).
  $$
- *Dilution* (sub-additivity for fully noisy extension): if a channel is extended by an
  independent fully noisy channel ($U^m$ with chance level $U^m$),
  $$
  D(P^n * U^m,\; Q^n * U^m) = \frac{D(P^n, Q^n) + D(U^m, U^m)}{m}.
  $$

### 1.3 Formal statement (from the supplement)

Preliminary definitions. $J = (0,1)$, $\hat J = (0,1]$, $\bar J = [0,1]$,
$$
K = (\bar J \times \bar J) \setminus (\hat J \times \{0\}),
$$
(i.e. all $(p,q) \in [0,1]^2$ except those with $p > 0, q = 0$ — the informativeness
would diverge there),
$$
\Gamma^n = \Big\{ X^n = (x_1,\ldots,x_n): x_\alpha \in \bar J,\ \sum_\alpha x_\alpha = 1 \Big\},
\qquad
\Delta^n = \{ (X^n, Y^n) \in \Gamma^n \times \Gamma^n : (x_\alpha, y_\alpha) \in K\ \forall \alpha \}.
$$

- **Property I (Causality).** $D: \Delta^n \to \mathbb{R}$ with
  $D(P^n, Q^n) = 0 \iff P^n = Q^n$.
- **Property II (Specificity).** $D(P^n, Q^n) := \max_\alpha \{ f(p_\alpha, q_\alpha) \}$
  for some $f: K \to \mathbb{R}$ that is continuous on $K$, analytic on
  $\hat J \times J$, with $f(0, q)$ analytic on $J$.
- **Property III (Intrinsicality).** For $(P^l, Q^l) \in \Delta^l$,
  $(P^m, Q^m) \in \Delta^m$:
  (a) expansion: $D(V^l * V^m, P^l * Q^m) = D(V^l, P^l) + D(V^m, Q^m)$;
  (b) dilution: $D(P^l * U^m, Q^l * U^m) = \big(D(P^l, Q^l) + D(U^m, U^m)\big)/m$.

**Theorem 1 (uniqueness).** If $D: \Delta^n \to \mathbb{R}$ ($n \ge 2$) satisfies
Properties I, II, III, then, uniquely up to a multiplicative constant $k > 0$,

$$
\boxed{\;
D(P^n, Q^n) = \max_\alpha \Big\{ k\, p_\alpha \log \frac{p_\alpha}{q_\alpha} \Big\},
\qquad f(p,q) = k\, p \log\frac{p}{q},\ (p,q) \in K
\;}
$$

with continuous extension at the boundary: $f(0,q) = 0$ for all $q$ (including
$f(0,0) = 0$), $f(1,q) = k\log(1/q)$, $f(p,1) = k\,p\log p$, $f(p,p) = 0$.
Set $k = 1$; the result is the **intrinsic difference (ID)**, in ibits.

Proof structure (supplement): show the form satisfies I–III; then show it is the *only*
solution given four technical assumptions (A1: $f(1, p'q')$ is a strict maximum
somewhere; A2: $\exists q_1$ with $f(1,q_1) > f(0, 1-q_1)$, which also forces $k>0$;
A3/A4: the "wrong branch" of the max is a strict maximum only finitely often); and
finally show that violating any of A1–A4 contradicts Property I or II (via a Pexider-type
functional equation $f(pq) = \max\{g(p),h(p)\} + \max\{g(q),h(q)\} \Rightarrow
f(x) = c\log x + d$, plus real-analytic continuation). A footnoted variant: if the max is
taken over $|f|$, the unique solution is $f(p,q) = k' p \log(p/q)$ with $k' \ne 0$.

### 1.4 Informativeness × selectivity decomposition

$$
\mathrm{ID}(P, Q) = \max_\alpha \underbrace{p_\alpha}_{\text{selectivity}}
\cdot \underbrace{\log\frac{p_\alpha}{q_\alpha}}_{\text{informativeness}} .
$$

- **Informativeness** $\log(p_\alpha/q_\alpha)$: the point-wise (Hartley/PMI-style)
  information of state $\alpha$ — deviation from chance. Additive over independent
  noiseless components (*expansion*: with $N$ fully specified units and uniform chance,
  informativeness $= N$ bits).
- **Selectivity** $p_\alpha$: how much probability mass is concentrated on that specific
  state. Multiplying by $p_\alpha < 1$ makes the measure sub-additive (*dilution*):
  adding an unconstrained unit to a fully specified one makes intrinsic information
  decrease **exponentially**, not stay constant.

Analogy in the paper: KLD $= \sum_\alpha p_\alpha \log(p_\alpha/q_\alpha)$ computes the
total "mass" of information; ID finds the point of highest information **density**.

### 1.5 Why ID replaces KLD (IIT 2.0) and EMD (IIT 3.0)

- **KLD** satisfies Causality but neither Specificity (it averages over states) nor
  Intrinsicality (it is additive for *all* independent distributions, so a channel
  extended by pure noise never loses information — it reflects what an extrinsic
  channel designer could recover with error-correcting codes, i.e., channel capacity).
  From the intrinsic perspective there is no encoder/decoder to agree on codes; a
  symbol must be taken as is. Example: 1 noiseless wire + 7 fully noisy wires: KLD = 1
  bit; ID ≈ 0 ibits.
- **EMD** (3.0) was a distance between whole repertoires; it neither selects a specific
  state (violates specificity/information postulate: "the experience is *this one*")
  nor satisfies the expansion/dilution behavior; nor is it unique — 4.0 explicitly
  requires measures that satisfy the postulates *uniquely* ("because consciousness is
  the way it is, the formulation … should be unique").
- ID is the **unique** measure satisfying causality (existence postulate),
  intrinsicality (intrinsicality postulate: expansion vs. dilution), and specificity
  (information postulate).
- Contrast with axiomatizations of KL (monotonicity, continuity, additivity) and Rényi
  (Kolmogorov–Nagumo means; the $\infty$-limit is state-specific but still always
  additive, hence lacks dilution). ID is sub-additive for *some* independent
  distributions (any $p_\alpha < 1$) and additive for others ($p_\alpha = 1$) — unusual
  among information measures.
- ID is smooth in $(P, Q)$ away from argmax ties and the $q=0$ boundary; the max makes
  it piecewise-smooth overall.

### 1.6 Behavior (worth reproducing as library tests)

- Noiseless 1-wire channel, uniform chance: ID = 1 ibit; noiseless 8-wire: 8 ibits
  (matches KL: 1 and 8 bits).
- 1 noiseless + 7 iid noisy wires with per-wire correct-transmission probability $s$:
  $p_\alpha = s^7$ on the $2^7$ compatible states, else 0; $Q = U^{2^8}$. ID goes from
  ~8 ibits ($s=1$) to ~0 ($s=1/2$); crosses 1 ibit at $s \approx 0.78$. KL stays ≥ 1 bit.
- $N$ iid wires with fixed $s = 0.88$, $Q = U$: $p_\alpha = s^{a_\alpha}(1-s)^{N-a_\alpha}$
  where $a_\alpha$ = number of agreeing bits; for $s \ge 0.5$ the max is at the sent
  symbol, ID $= s^N \cdot N \log(2s)$ — informativeness grows linearly
  ($N\log(r/[1/2])$), selectivity decays exponentially ($r^N$), product peaks at a
  finite optimal $N$ (here $N = 8$, 2.41 ibits). KL increases monotonically in $N$.
- Network-element example (sender side): $P^n = \bigotimes_j P(Y_j \mid X = 1^N)$
  (conditionally independent outputs), $Q^n = P(Y) = \sum_x P(Y|x)P(x)$ with uniform
  $P(x)$ — note $Q$ here is **not** uniform and not a product (outputs share inputs, so
  they are correlated in the joint), and ID then differs per observed symbol.

---

## 2. IIT 4.0: substrate, TPM conventions

### 2.1 Substrate and TPM

A substrate is a stochastic system $U = \{U_1, \ldots, U_n\}$ of $n$ interacting units,
state space $\Omega_U = \prod_i \Omega_{U_i}$ (finite), current state $u \in \Omega_U$.
States are represented as sets of tuples $u = \{(U_i, \mathrm{state}(U_i))\}$ so set
operations respect unit identity *and* state.

The complete transition probability function over one discrete update $u \to \bar u$:

$$
\mathcal{T}_U \equiv p(\bar u \mid u), \qquad u, \bar u \in \Omega_U .
\tag{1}
$$

**Assumptions** (all load-bearing for the formalism):

1. Discrete updates; finite state space.
2. **Conditional independence** of units given the preceding full state:
   $$
   p(\bar u \mid u) = \prod_{i=1}^n p(\bar u_i \mid u).
   \tag{2}
   $$
3. **Interventional completeness**: $p(\bar u \mid u) = p(\bar u \mid do(u))$ for every
   state — $\mathcal{T}_U$ is obtained under Pearl's do-operator, so $U$ is a causal
   network and $\mathcal{T}_U$ is a $|\Omega_U| \times |\Omega_U|$ TPM describing the
   system's evolution under **all possible interventions**, not observed statistics.
4. Probabilities relevant to intrinsic existence are always computed by imposing a
   **uniform** distribution over possible (intervened) states — never the observed
   stationary distribution (intrinsic-perspective requirement, Box 1).

Convention in the paper's examples: binary units $\Omega_{U_i} = \{-1, +1\}$, uppercase
letters = state +1/ON, lowercase = −1/OFF; example dynamics are logistic units
$$
p(U_{i,t} = 1 \mid u_{t-1}) = \frac{1}{1 + \exp(-k \sum_j w_{j,i} u_{j,t-1})},
\qquad \sum_j w_{j,i} = 1,
\tag{60–61}
$$
with $k$ the determinism (inverse noise) parameter.

### 2.2 Candidate systems and background conditions (intrinsicality)

A candidate system is $S \subseteq U$ in state $s \subseteq u$; the background is
$W = U \setminus S$ in state $w = u \setminus s$. Background units are **causally
marginalized, conditional on the current state of the universe** — rendered causally
inert. This is done per system-unit and recombined as a product (preserving conditional
independence and killing residual background-induced correlations). Two different
system TPMs result:

**Effect TPM** — background fixed at its *current* state $w$ (the state of $W$ at the
time of the transition $s \to \bar s$ is known from $u$):
$$
\mathcal{T}_e \equiv p_e(\bar s \mid s) = p(\bar s \mid s, w),
\qquad s, \bar s \in \Omega_S .
\tag{3}
$$

**Cause TPM** — the *prior* background state $\bar w$ is unknown; it is assigned its
posterior given the current universe state $u$ (uniform prior over full prior states),
and each system unit's conditional is averaged under that posterior, then multiplied:
$$
\mathcal{T}_c \equiv p_c(s \mid \bar s)
= \prod_{i=1}^{|S|} \sum_{\bar w} p(s_i \mid \bar s, \bar w)\,
\underbrace{\frac{\sum_{\hat s} p(u \mid \hat s, \bar w)}{\sum_{\hat u} p(u \mid \hat u)}}_{p(\bar w \mid u)},
\qquad s, \bar s \in \Omega_S .
\tag{4}
$$

History of this choice (S2 Text): 3.0 fixed background at current state for effects and
at the *actual past* state for causes; post-3.0 papers fixed it at the current state for
both (leading to unreachable current states / missing causes); 4.0's causal
marginalization conditional on $u$ avoids unreachable states and needs only the current
background state ("context").

---

## 3. System level: maximal cause–effect state and φ_s

### 3.1 System intrinsic information ii

For system $S$ in current state $s$ over a candidate effect state $\bar s$:
$$
ii_e(s, \bar s) = p_e(\bar s \mid s)\, \log\!\left( \frac{p_e(\bar s \mid s)}{p_e(\bar s)} \right),
\tag{5}
$$
with the **unconstrained effect probability** (uniform average over interventions on
current states):
$$
p_e(\bar s) = |\Omega_S|^{-1} \sum_{s \in \Omega_S} p_e(\bar s \mid s).
\tag{6}
$$

Over a candidate cause state $\bar s$:
$$
ii_c(s, \bar s) = p_c^{\leftarrow}(\bar s \mid s)\,
\log\!\left( \frac{p_c(s \mid \bar s)}{p_c(s)} \right),
\tag{7}
$$
with the unconstrained probability of the current state
$$
p_c(s) = |\Omega_S|^{-1} \sum_{\bar s \in \Omega_S} p_c(s \mid \bar s),
\tag{8}
$$
and the **backward (Bayes) probability** with uniform prior over cause states:
$$
p_c^{\leftarrow}(\bar s \mid s)
= \frac{p_c(s \mid \bar s)\, |\Omega_S|^{-1}}{p_c(s)}
= \frac{p_c(s \mid \bar s)}{\sum_{\hat s \in \Omega_S} p_c(s \mid \hat s)}.
\tag{9}
$$

Anatomy: the log term is the **informativeness** — expressed with *forward*
probabilities on both sides ($ii_e$: increase of the effect state's probability due to
$s$; $ii_c$: increase of $s$'s probability due to the cause state — existence requires
both $\log(p_e(\bar s|s)/p_e(\bar s)) > 0$ (10) and $\log(p_c(s|\bar s)/p_c(s)) > 0$
(11)). The prefactor is the **selectivity** — the forward conditional on the effect side
but the *backward* conditional $p_c^{\leftarrow}$ on the cause side. Consequently
$ii_e$ is *exactly* the ID between constrained and unconstrained effect repertoires,
while $ii_c$ is **not** literally an ID between two distributions (mixed
backward-selectivity/forward-informativeness) — it is defined by analogy as
selectivity × informativeness. (Changed relative to Barbosa 2021, which used backward
probabilities in the informativeness too; 4.0 uses forward informativeness on the cause
side to comply with the existence postulate. Also, no absolute value inside ii —
mechanisms must *raise* the relevant probabilities.)

Properties: $ii_e(\mathcal{T}_e, s) = 0 \iff p_e(\bar s|s) = p_e(\bar s)\ \forall \bar s$
(no causal power); indeterminism lowers effect selectivity; degeneracy (many states
mapping into one) lowers cause selectivity even in deterministic systems.

### 3.2 Maximal cause–effect state (information postulate)

$$
s'_e(\mathcal{T}_e, s) = \arg\max_{\bar s \in \Omega_S} ii_e(s, \bar s),
\qquad
ii_e(\mathcal{T}_e, s) := \max_{\bar s} ii_e(s, \bar s),
\tag{12–13}
$$
and analogously $s'_c = \arg\max_{\bar s} ii_c(s,\bar s)$. The pair
$s' = \{s'_c, s'_e\}$ is the **maximal cause–effect state** of the system. (This is the
principle of maximal existence at the state level.) $s'$ need not equal the actual
past/future states of the dynamics.

Tie rule (see §8): among states tied in $ii$, pick the one with maximal
$\varphi_s(\mathcal{T}_e,\mathcal{T}_c,s,\theta')$; remaining ties are resolved at the
level of $\Phi$.

### 3.3 Directional system partitions Θ(S) (integration)

A partition $\theta \in \Theta(S)$ divides $S$ into $k \ge 2$ parts:
$$
\theta = \{ S^{(1)}_{\delta_1}, S^{(2)}_{\delta_2}, \ldots, S^{(k)}_{\delta_k} \},
\qquad
S^{(i)} \ne \emptyset,\quad S^{(i)} \cap S^{(j)} = \emptyset,\quad
\bigcup_i S^{(i)} = S,
\tag{14–15}
$$
where each part carries a **direction flag** $\delta_i \in \{\leftarrow, \rightarrow,
\leftrightarrow\}$: the part's inputs (←), outputs (→), or both (↔) are cut (replaced by
independent noise). Rationale: to be a part of a system a subset must interact with the
rest in *both* directions; directional cuts detect parts that interact only weakly or
one-way ("fault lines").

For each part $S^{(i)}$, the set of units whose **inputs to $S^{(i)}$ are cut**:
$$
X^{(i)} =
\begin{cases}
S \setminus S^{(i)} & \text{if } \delta_i \in \{\leftarrow, \leftrightarrow\} \\[2pt]
\bigcup_{j \ne i:\ \delta_j \in \{\rightarrow, \leftrightarrow\}} S^{(j)} & \text{if } \delta_i = \rightarrow,
\end{cases}
\tag{16}
$$
and $Y^{(i)} = S \setminus X^{(i)}$ (inputs left intact). The partitioned TPM noises all
cut connections, unit-wise (product form preserves conditional independence):
$$
\mathcal{T}_e^\theta \equiv p_e^\theta(\bar s \mid s)
= \prod_{j=1}^{n} p_e^\theta(\bar s_j \mid s),
\qquad
p_e^\theta(\bar s_j \mid s)
= |\Omega_{X^{(i)}}|^{-1} \sum_{x^{(i)} \in \Omega_{X^{(i)}}} p_e(\bar s_j \mid x^{(i)}, y^{(i)}),
\tag{17–18}
$$
for $S_j \in S^{(i)}$, with $y^{(i)} = s \setminus x^{(i)}$; analogously
$\mathcal{T}_c^\theta$.

### 3.4 System integrated information φ_s

Loss of intrinsic information about the maximal states due to the partition
(same *form* as ii, with the partitioned probability replacing the unconstrained one):
$$
\varphi_e(\mathcal{T}_e, s, \theta)
= p_e(s'_e \mid s)\,
\left| \log\!\left( \frac{p_e(s'_e \mid s)}{p_e^\theta(s'_e \mid s)} \right) \right|_+,
\tag{19}
$$
$$
\varphi_c(\mathcal{T}_c, s, \theta)
= p_c^{\leftarrow}(s'_c \mid s)\,
\left| \log\!\left( \frac{p_c(s \mid s'_c)}{p_c^\theta(s \mid s'_c)} \right) \right|_+.
\tag{20}
$$

Principle of minimal existence twice over:
$$
\varphi_s(\mathcal{T}_e, \mathcal{T}_c, s, \theta)
= \min\{ \varphi_c(\mathcal{T}_c, s, \theta),\ \varphi_e(\mathcal{T}_e, s, \theta) \},
\tag{21}
$$
$$
\varphi_s(\mathcal{T}_e, \mathcal{T}_c, s) := \varphi_s(\mathcal{T}_e, \mathcal{T}_c, s, \theta'),
\tag{22}
$$
with the **minimum partition (MIP)** selected under normalization by the maximum
possible value over arbitrary TPMs of the same shape:
$$
\theta' = \arg\min_{\theta \in \Theta(S)}
\frac{\varphi_s(\mathcal{T}_e, \mathcal{T}_c, s, \theta)}
     {\max_{\mathcal{T}'_e, \mathcal{T}'_c} \varphi_s(\mathcal{T}'_e, \mathcal{T}'_c, s, \theta)},
\qquad
\max_{\mathcal{T}'_e, \mathcal{T}'_c} \varphi_s(\cdot, \theta)
= \sum_{i=1}^{k} |S^{(i)}|\, |X^{(i)}| ,
\tag{23}
$$
i.e. normalization = maximal number of pairwise "connections" severed by $\theta$
(proof in Marshall et al. 2023). The *reported* $\varphi_s$ across the MIP is the
**unnormalized** value (an absolute quantity); normalization is used only to *select*
$\theta'$. If several $\theta$ tie in normalized value, choose the one with the largest
**unnormalized** $\varphi_s$ (maximal existence). Consequences: the MIP finds genuine
fault lines (e.g. cuts across a "bridge") instead of defaulting to single-unit cuts;
$\varphi_s = 0$ whenever the system's (effective) graph is not strongly connected —
purely feed-forward systems have $\varphi_s = 0$.

### 3.5 Exclusion: complexes and the condensation of the universe

**Maximal substrate (complex)**: a candidate system with $\varphi_s$ maximal compared to
all candidate systems with overlapping units:
$$
S \cap \tilde S \ne \emptyset \implies \varphi_s(s) > \varphi_s(\tilde s)
\quad \forall\, \tilde S \ne S \subseteq U .
\tag{26}
$$

Recursive search within a universal substrate $U_0$ in state $u_0$:
$$
\varphi_s^{*}(\mathcal{T}_e, \mathcal{T}_c, u_k) = \max_{S \subseteq U_k} \varphi_s(\mathcal{T}_e, \mathcal{T}_c, s),
\qquad
S_k^{*} = \arg\max_{S \subseteq U_k} \varphi_s(\mathcal{T}_e, \mathcal{T}_c, s),
\tag{24–25}
$$
then $U_{k+1} = U_k \setminus S_k^{*}$, repeated until $U_{k+1} = \emptyset$ or
$U_{k+1} = U_k$. Excluded units remain background conditions. The universe thereby
"condenses" into a disjoint, exhaustive set of complexes (first complex, second complex,
…; single-unit complexes are "monads"). If the $\arg\max$ ties among *overlapping*
systems: compare their $\Phi$; if still tied, those systems fail exclusion and the next
best *unique* system is taken (S1 Text).

### 3.6 Unit grains (macro units)

All unit/update/state grains must in principle be considered. A macro unit $J = j$ over
micro substrate $\hat S \subseteq U$ is defined by a state mapping
$$
j = g(\hat s), \qquad g: \Omega_{\hat S} \to \Omega_J .
$$
Constraints: a candidate unit must itself be **maximally irreducible within** — its
$\varphi_s$ as a candidate system must exceed that of any system definable from a subset
of $\hat S$ (units must not "disintegrate"); but units need only be maximal *within*
(they are not complexes). Among all valid candidate-unit sets, the winning grain is the
one that maximizes the $\varphi_s$ of the **complex the units compose** (maximal
existence at the system level, not the unit level). The search is iterative
(micro → meso → macro). Details of $g$ and macro-TPM derivation are deferred to
Marshall, Findlay, Albantakis, Tononi, *A Mathematical Framework for Cause-Effect Power
Analysis of Macro Units* (in prep. at publication time; footnote 11).

---

## 4. Mechanism level: causal distinctions and φ_d

### 4.1 Definitions

Within a complex $S$ (state $s$), a **mechanism** is a subset $M \subseteq S$ in state
$m \subseteq s$ that irreducibly links a cause state and an effect state over
**purviews** $Z_c, Z_e \subseteq S$. A **candidate distinction** is the tuple
$$
d(m) = (m, z^{*}, \varphi_d), \qquad z^{*} = \{ z_c^{*}, z_e^{*} \},\ \varphi_d > 0 .
\tag{27}
$$

### 4.2 Purview probabilities (causal marginalization + product probabilities)

Everything is computed from $\mathcal{T}_e$, $\mathcal{T}_c$ (so background units are
already handled). Units in $X = S \setminus M$ (non-mechanism system units, when
conditioning on the mechanism) are causally marginalized with a **uniform**
distribution (unlike background units, which are marginalized conditional on $u$ —
footnote 12):
$$
p_e(z_i \mid m) = |\Omega_X|^{-1} \sum_{x \in \Omega_X} p(z_i \mid m, x),
\qquad z_i \in \Omega_{Z_i}.
\tag{28}
$$

Joint purview probabilities are **product probabilities** over single-unit terms —
this deletes correlations induced by common inputs from $X$ that are not effects of $M$
(3.0 called this mechanism "virtual elements"; 4.0 formalizes as products):
$$
\pi_e(z \mid m) = \prod_{i=1}^{|Z|} p_e(z_i \mid m),
\qquad
\pi_c(m \mid z) = \prod_{i=1}^{|M|} p_c(m_i \mid z),
\tag{29–30}
$$
where on the cause side, $Y = S \setminus Z$ is uniformly marginalized:
$p_c(m_i \mid z) = |\Omega_Y|^{-1} \sum_{y \in \Omega_Y} p_c(m_i \mid z, y)$.

Unconstrained versions (uniform average over the conditioning variable):
$$
\pi_e(z; M) = |\Omega_M|^{-1} \sum_{m \in \Omega_M} \pi_e(z \mid m),
\qquad
\pi_c(m; Z) = |\Omega_Z|^{-1} \sum_{z \in \Omega_Z} \pi_c(m \mid z).
\tag{31–32}
$$

Backward cause probability (Bayes with uniform prior over purview states):
$$
\pi_c^{\leftarrow}(z \mid m)
= \frac{\pi_c(m \mid z)\, |\Omega_Z|^{-1}}{\pi_c(m; Z)}
= \frac{\prod_i p_c(m_i \mid z)}{\sum_{\hat z \in \Omega_Z} \prod_i p_c(m_i \mid \hat z)} .
\tag{33}
$$

### 4.3 Mechanism intrinsic information and state selection

$$
ii_e(m, z) = \pi_e(z \mid m) \log\!\left( \frac{\pi_e(z \mid m)}{\pi_e(z; M)} \right),
\qquad
ii_c(m, z) = \pi_c^{\leftarrow}(z \mid m) \log\!\left( \frac{\pi_c(m \mid z)}{\pi_c(m; Z)} \right).
\tag{34–35}
$$

For each candidate purview $Z$, the specific state the mechanism selects:
$$
z'_e(m, Z) = \arg\max_{z \in \Omega_Z} ii_e(m, z),
\qquad
ii_e(m, Z) := \max_z ii_e(m, z),
\tag{36–37}
$$
(analogously $z'_c$). Selection order matters and is deliberate (S2 Text): the state is
chosen by **ii first**, *before* any partitioning ("the cause and effect of $m$ should
be determined by the mechanism as a whole, independent of how it can be partitioned") —
a change relative to Barbosa 2021, which selected states by φ.

### 4.4 Disintegrating mechanism partitions Θ(M, Z)

$$
\Theta(M, Z) = \Big\{ \{ (M^{(i)}, Z^{(i)}) \}_{i=1}^{k} :\ k \in \{2, 3, \ldots\},\
M^{(i)} \in \mathcal{P}(M),\ Z^{(i)} \in \mathcal{P}(Z),
$$
$$
\textstyle\bigcup_i M^{(i)} = M,\ \bigcup_i Z^{(i)} = Z,\
Z^{(i)} \cap Z^{(j)} = M^{(i)} \cap M^{(j)} = \emptyset\ \forall i \ne j,\
M^{(i)} = M \Rightarrow Z^{(i)} = \emptyset \Big\}.
\tag{38}
$$

That is: **disjoint (possibly empty) parts** that jointly cover both $M$ and $Z$, pairing
a mechanism-part with a purview-part; the final clause bars the trivial "partition" that
keeps the whole mechanism with the whole purview — any part containing all of $M$ must
have an empty purview part. Effect: a disintegrating partition either cuts the mechanism
into ≥ 2 independent parts (possible when $|M| > 1$), or severs **all** connections
between $M$ and $Z$ (forced when $|M| = 1$). (Introduced in Albantakis 2019 / Barbosa
2021; sometimes described as the set of disjoint "tri-partition"-style cuts
$\{(M^{(1)},Z^{(1)}), (M^{(2)},Z^{(2)}), \ldots\}$ with empty parts allowed.) These
differ from system partitions: mechanism–purview pairs are already directed, so no
direction flags.

Partitioned probability (product over parts, with conventions
$\pi(\emptyset \mid m^{(i)}) = \pi(\emptyset) = 1$, and $m^{(i)} = \emptyset$ meaning
full causal marginalization of the mechanism):
$$
\pi_e^\theta(z'_e \mid m) = \prod_{i=1}^{k} \pi_e\big( z'^{(i)}_e \mid m^{(i)} \big),
\qquad
\pi_e(z \mid \emptyset) = \prod_{i=1}^{|Z|} |\Omega_S|^{-1} \sum_{s \in \Omega_S} p_e(z_i \mid s).
\tag{39–40}
$$

### 4.5 Mechanism integrated information and the distinction MIP

$$
\varphi_e(m, Z, \theta)
= \pi_e(z'_e \mid m)
\left| \log\!\left( \frac{\pi_e(z'_e \mid m)}{\pi_e^\theta(z'_e \mid m)} \right) \right|_+ ,
\tag{41}
$$
$$
\varphi_e(m, Z) := \varphi_e(m, Z, \theta'),
\qquad
\theta' = \arg\min_{\theta \in \Theta(M, Z)}
\frac{\varphi(m, Z, \theta)}{\max_{\mathcal{T}'} \varphi(m, Z, \theta)} ,
\tag{42–43}
$$
again normalized by the partition's maximum possible value over arbitrary TPMs (= the
number of possible pairwise interactions affected by the partition; exact expression
given in Barbosa 2021, not restated in the 4.0 paper). On the cause side:
$$
\varphi_c(m, Z) := \varphi_c(m, Z, \theta')
= \pi_c^{\leftarrow}(z'_c \mid m)
\left| \log\!\left( \frac{\pi_c(m \mid z'_c)}{\pi_c^{\theta'}(m \mid z'_c)} \right) \right|_+ .
\tag{44}
$$

### 4.6 Purview selection (exclusion at the mechanism level) and φ_d

The mechanism's definite effect (cause) purview + state is the maximally irreducible
one over **all subsets** $Z \subseteq S$:
$$
z_e^{*}(m) = \arg\max_{Z \subseteq S} \varphi_e\big( m, z'_e(m, Z) \big),
\qquad
\varphi_e(m) := \max_{Z \subseteq S} \varphi_e\big( m, z'_e(m, Z) \big),
\tag{45–46}
$$
analogously $z_c^{*}(m)$, $\varphi_c(m)$. The distinction's irreducibility (principle of
minimal existence):
$$
\varphi_d(m) = \min\big( \varphi_c(m),\ \varphi_e(m) \big).
\tag{47}
$$

### 4.7 Congruence and the distinction set D

Distinctions must be **congruent** with the system's maximal cause–effect state
$s' = \{s'_c, s'_e\}$ (information postulate at the system level; new in 4.0):
$$
D(\mathcal{T}_e, \mathcal{T}_c, s)
= \{ d(m) : m \subseteq s,\ \varphi_d(m) > 0,\
z_c^{*}(m) \subseteq s'_c,\ z_e^{*}(m) \subseteq s'_e \}.
\tag{48}
$$
(⊆ over unit–state tuples: purview units must be specified *in the same state* as in
the system's cause/effect state.) Incongruent candidate distinctions do not exist for
the complex. Upper bound: $|D| \le 2^n - 1$ for an $n$-unit complex (all non-empty
mechanisms).

---

## 5. Relations and φ_r

### 5.1 Definitions

A set of distinctions $\mathbf{d} \subseteq D(s)$, $|\mathbf{d}| \ge 1$, is **related**
if the causes/effects of its members overlap congruently — over the same units in the
same state. Writing $z_c^{*}(d), z_e^{*}(d)$ for the cause/effect of distinction $d$, a
"relating" set of purviews picks the cause, the effect, or both from each distinction:
$$
\mathbf{z} :\ \mathbf{z} \cap \{ z_c^{*}(d), z_e^{*}(d) \} \ne \emptyset\ \ \forall d \in \mathbf{d},
\qquad
\bigcap_{z \in \mathbf{z}} z \ne \emptyset,
\qquad |\mathbf{z}| > 1,
\tag{49}
$$
with **maximal (face) overlap**
$$
o^{*}(\mathbf{z}) = \bigcap_{z \in \mathbf{z}} z \ne \emptyset
\tag{50}
$$
(intersection over unit–state tuples — congruence is built in). Each valid $\mathbf{z}$
is a **face** $f(\mathbf{z}) = (\mathbf{z}, o^{*}(\mathbf{z}))$ (52); a face over
$k = |\mathbf{z}|$ purviews is a $k$-degree face; a relation over $h = |\mathbf{d}|$
distinctions is an $h$-degree relation, with up to $3^{|\mathbf{d}|}$ faces
(cause / effect / both, per distinction). The case
$\mathbf{z} = \{z_c^{*}(d), z_e^{*}(d)\}$ for a single distinction is a **self-relation**.

A relation is the tuple
$$
r(\mathbf{d}) = (\mathbf{d}, \mathbf{f}(\mathbf{d}), \varphi_r), \qquad \varphi_r > 0 .
\tag{51}
$$

### 5.2 Relation integrated information φ_r

No re-partitioning of mechanisms: distinctions are already-established irreducible
components, so a distinction contributes its whole $\varphi_d$ **density**. Assume
$\varphi_d$ is uniformly distributed over the distinction's distinct purview units
(union over unit–state tuples, so cause/effect units congruent with each other count
once, incongruent ones separately):
$$
\text{density of } d = \frac{\varphi_d}{| z_c^{*}(d) \cup z_e^{*}(d) |}.
\tag{53}
$$
The **relation purview** (joint purview) is the union of all face overlaps
$\big| \bigcup_{f \in \mathbf{f}(\mathbf{d})} o_f^{*} \big|$ (54). Then, by minimal
existence (a relation is only as irreducible as its weakest distinction's contribution):
$$
\boxed{\;
\varphi_r(\mathbf{d})
= \min_{d \in \mathbf{d}}\;
\Big| \bigcup_{f \in \mathbf{f}(\mathbf{d})} o_f^{*} \Big|\;
\frac{\varphi_d}{| z_c^{*}(d) \cup z_e^{*}(d) |}
\;}
\tag{55}
$$
Guarantees: $\varphi_r \le \min_d \varphi_d$, with equality iff every distinction's
cause and effect are fully overlapped by all others. Exclusion is automatic: taking
sub-overlaps could only lower $\varphi_r$, so maximal overlaps are used.

Set of relations over a distinction set:
$$
R(D) = \{ r(\mathbf{d}) : \varphi_r(\mathbf{d}) > 0 \},\quad \forall \mathbf{d} \subseteq D .
\tag{56}
$$
Upper bound on count: $2^{(2^n - 1)} - 1$.

### 5.3 Analytical results (S3 Text) — essential for implementation

Let $Z(\mathcal{T}_e,\mathcal{T}_c,s) = \{ (z_d, \varphi_d) \}$ with
$z_d = z_c^{*}(d) \cup z_e^{*}(d)$ for each $d \in D$.

Key identity (proved in S3): the relation purview equals the intersection of the
distinctions' cause∪effect purviews,
$$
\bigcup_{f \in \mathbf{f}(\mathbf{d})} o_f^{*}
= \bigcap_{d \in \mathbf{d}} \big( z_c^{*}(d) \cup z_e^{*}(d) \big),
$$
so
$$
\varphi_r(\mathbf{d}) = \Big| \bigcap_{d \in \mathbf{d}} z_d \Big|\;
\min_{d \in \mathbf{d}} \frac{\varphi_d}{|z_d|}.
$$

**Sum of φ_r without enumerating relations.** For a unit–state $n$, let
$Z(n) = \{ (z, \varphi) \in Z : n \in z \}$. Sort $Z(n)$ by $\varphi/|z|$
non-decreasing: $(z_{(1)}, \varphi_{(1)}), (z_{(2)}, \varphi_{(2)}), \ldots$ Then
(excluding self-relations, which are handled individually — there are only $|D|$):
$$
\sum_{\substack{\mathbf{r} \subseteq Z,\ |\mathbf{r}| \ge 2}}
\Big| \bigcap_{(z,\varphi) \in \mathbf{r}} z \Big|
\min_{(z,\varphi) \in \mathbf{r}} \frac{\varphi}{|z|}
= \sum_{n \in s'_c \cup s'_e}\;
\sum_{j=1}^{|Z(n)|} \frac{\varphi_{(j)}}{|z_{(j)}|} \big( 2^{|Z(n)| - j} - 1 \big).
$$
(The per-unit factoring over-counts each relation exactly $|\bigcap z|$ times, cancelling
the purview-size factor; $2^{|Z(n)|-j} - 1$ counts subsets whose minimum is the $j$-th
element.)

**Number of relations** (excluding self-relations), with
$Z(o) = \{ (z,\varphi) \in Z : z \supseteq o \}$ for $o \subseteq s'_c \cup s'_e$,
via inclusion–exclusion:
$$
\#R = \sum_{\emptyset \neq o \subseteq s'_c \cup s'_e} (-1)^{|o| - 1}
\big( 2^{|Z(o)|} - |Z(o)| - 1 \big).
$$

---

## 6. Φ-structures and big Phi

Cause–effect structure:
$$
C(D) = D \cup R(D).
\tag{57}
$$
The cause–effect structure of a **complex** is the **Φ-structure**
$C(\mathcal{T}_e, \mathcal{T}_c, s^{*})$ (58). Structure integrated information:
$$
\boxed{\;
\Phi(\mathcal{T}_e, \mathcal{T}_c, s^{*})
= \sum_{C(\mathcal{T}_e, \mathcal{T}_c, s^{*})} \varphi
= \sum_{d \in D} \varphi_d + \sum_{r \in R(D)} \varphi_r
\;}
\tag{59}
$$

**Φ in 4.0 is a sum, not a partitioned distance.** Explicitly (S2 Text): "Φ is not
computed based on a partition (as system phi), but rather a sum of the integrated
information within the structure (where each term of the sum was computed by
partitioning)"; there is no normalization in Φ (unlike the φ_s MIP search); the sum was
"chosen as the simplest option that captures all that exists within the complex."
Quality of experience = the Φ-structure; quantity = Φ. Whether a system exists as one
entity is decided by $\varphi_s$ (non-compositional); how much it exists as a structure
is Φ.

**Φ-folds** (sub-structures): a Φ-fold is any subset of the distinctions and relations
of a Φ-structure. Named cases: the *distinction Φ-fold* $C(\{d\})$ = one distinction
plus all relations bound to it (its context); a *compound Φ-fold* = the distinction
Φ-folds specified by a subset of units; a *content Φ-fold* = a highly interrelated
subset of distinctions. Footnote 13 gives a partition of Φ over distinction Φ-folds:
assign each $d \in \mathbf{d}$ the share $\varphi_r(\mathbf{d})/|\mathbf{d}|$; the
resulting per-distinction $\Phi_d$ values sum to Φ.

Empirical anchors from the paper's examples (useful regression tests; all with logistic
units, $k = 4$, states as in Fig 6/7/8): degenerate bottleneck network → complex $Ab$
plus 4 monads; modular network → 3 two-unit complexes; 6-unit copy cycle →
$\varphi_s = 1.74$ ibits but only first-order distinctions, $\Phi = 7.65$; 6-unit
specialized lattice → 27 of 63 possible distinctions, > 1.5 × 10⁶ relations,
$\Phi = 11452$ ibits; near-maximal lattice variant → 4-unit complex $Abef$
($\varphi_s = 0.27$) excludes the integrated 6-unit superset ($\varphi_s = 0.15$).
Fig 7: 5-unit net, state $ABcdE$: $\varphi_s = 1.1$, 23 distinctions, 13740 relations,
$\Phi = 22.26$; unit E inactive (OFF): same distinction count, different purviews/φ values, fewer
relations, somewhat lower Φ; unit E *inactivated* (no
counterfactual states): complex shrinks to $ABcd$, 14 distinctions, $\Phi = 3.35$.
Fig 8: three functionally equivalent 3-unit coin-counting systems: $\Phi = 21.01$ vs
$\Phi = 3.64$ (both $\varphi_s = 2$) vs reducible ($\varphi_s = 0$, feed-forward).

---

## 7. Tie resolution (S1 Text) — normative algorithm

Ties arise from TPM symmetries (common in small deterministic toy models; measure-zero
in realistic noisy systems). General rule: resolve by the principle of maximal
existence, consulting the *subsequent* postulates in order.

1. **System cause–effect state ties** (states tied in $ii_{c/e}$, Eq 12): choose the
   state with maximal $\varphi_s(\mathcal{T}_e, \mathcal{T}_c, s, \theta')$.
2. **Still tied in φ_s**: rarely matters for system selection, but for unfolding choose
   the state that maximizes the structure integrated information Φ. If two or more
   states also tie in Φ, the system **fails the information postulate and is not a
   complex** (unless the tied cause–effect structures are intrinsically identical, in
   which case the tie is merely extrinsic).
3. **System (complex) ties** (overlapping candidates tied for $\max \varphi_s$): choose
   the one with maximal Φ. If also tied in Φ: none of them complies with exclusion —
   none is a complex; choose the next-best system (by φ_s) that is unique.
4. **System MIP ties** (several partitions minimize normalized φ_s): take the partition
   with the largest **unnormalized** φ_s (stated in the main text after Eq 23).
5. **Mechanism state ties**: state chosen by $ii(m,z)$ first; all tied maximal states
   are carried into the φ comparison across purviews (Eq 45). Ties in
   $\max \varphi_d(m, Z)$ / in $z^{*}_{c/e}$ are resolved at the level of the
   cause–effect structure by choosing the $z^{*}_{c/e}$ that maximizes Φ. In practice:
   for state ties **within** the same purview, pick the state congruent with $s'$;
   for ties **across** purviews, the winner is generally the one supporting the most
   relations, which typically favors **larger purviews**.

---

## 8. Non-binary units, stochastic dynamics, conditional independence

- **The formalism is not restricted to binary units.** Everything is defined over
  arbitrary finite per-unit state spaces: $\Omega_U = \prod_i \Omega_{U_i}$;
  probabilities, partitions, marginalizations, ID, and all φ quantities are defined on
  general finite alphabets. Only the *examples* use binary units
  ($\Omega_{U_i} = \{-1, +1\}$, or $\{0,1\}$ in Fig 8) with logistic activation. The
  macro-unit framework in fact naturally produces multi-valued macro units
  ($g: \Omega_{\hat S} \to \Omega_J$ with $|\Omega_J|$ arbitrary).
- **Stochastic dynamics are the generic case.** $\mathcal{T}_U$ is a stochastic TPM;
  determinism is a special case. IIT 4.0 argues realistic substrates require some
  micro-indeterminism (S1 Text), and indeterminism/degeneracy quantitatively reduce
  selectivity, hence ii and φ.
- **Conditional independence (Eq 2) is an axiom-level modeling assumption**: units are
  conditionally independent given the *complete preceding* state, i.e. any observed
  same-time correlation must be attributable to common causes within the model. This is
  what legitimizes (a) all product-form constructions (Eqs 4, 17, 29, 30, 39) and
  (b) the interpretation of unit-wise noising as "physical" independent noise
  injection. It corresponds to units being observable/manipulable independently and
  "irreducible within." Footnote 7: extension to finite quantum systems under unitary
  evolution exists (Albantakis, Prentner, Durham 2023), where conditional independence
  applies to non-entangled subsystems.
- **Time and updates**: discrete update steps; one-step causes/effects only (grain of
  updates is itself subject to the exclusion postulate via the macro framework).
- **Interventional semantics**: all probabilities are do-probabilities; TPM must be the
  complete interventional description (a "causal network"). Observed frequencies are
  never used; the uniform interventional prior is mandated by the intrinsic
  perspective.

---

## 9. IIT 3.0 vs 4.0 — precise diff table

Authoritative source: S2 Text of the 4.0 paper (plus main text and Oizumi, Albantakis,
Tononi 2014 for 3.0 specifics).

| Aspect | IIT 3.0 (2014) | IIT 4.0 (2023) |
|---|---|---|
| Difference measure | Earth Mover's Distance (EMD) between repertoires; extended EMD between conceptual structures for Φ. (2.0 used KLD.) | Intrinsic Difference (ID): $\max_\alpha p_\alpha \log(p_\alpha/q_\alpha)$ — unique under causality/intrinsicality/specificity; selectivity × informativeness. |
| What a mechanism specifies | Whole cause/effect **repertoires** (probability distributions) | A **specific cause state and effect state** ($z^{*}_c, z^{*}_e$) selected by max ii |
| Cause-side computation | Cause repertoire via Bayes with uniform prior; EMD to unconstrained repertoire | Selectivity = backward $\pi_c^{\leftarrow}$ (Bayes, uniform prior); informativeness = **forward** ratio $\pi_c(m|z)/\pi_c(m;Z)$; no absolute value; $ii_c$ not literally an ID |
| Unconstrained comparison | Unconstrained (max-entropy) repertoires as EMD reference | Unconstrained probabilities = uniform-averaged marginals (Eqs 6, 8, 31, 32); ID reference distribution; partitioned probability replaces it inside φ |
| Background conditions | Frozen at current state for effects; at **actual past state** for causes | **Causally marginalized conditional on current universe state** (Eq 4); avoids unreachable current states; only current context needed. (Interim post-3.0 papers froze current state for both.) |
| Order of operations | Composition first: concept structure computed for every candidate set, complex = max Φ^max over sets | Integration/exclusion first: complexes found by **φ_s alone** (non-compositional, as in 2.0); Φ-structure unfolded only for the winning complex |
| System partitions | **Unidirectional bipartitions**: noise connections from one part to the other (one direction), no normalization | **Directional k-partitions** $\theta = \{S^{(i)}_{\delta_i}\}$, $\delta_i \in \{\leftarrow,\rightarrow,\leftrightarrow\}$, $k \ge 2$; MIP selected by φ_s **normalized** by $\sum_i |S^{(i)}||X^{(i)}|$ (max possible φ_s for the cut); reported value unnormalized. Partitions remain directional as in 3.0, but per-part direction flags + multi-part + normalization are new; sensitive to fault lines, does not default to single-unit cuts |
| System φ | Φ (of the conceptual structure) over the system MIP: XEMD between full and partitioned concept structures | $\varphi_s = \min(\varphi_c, \varphi_e)$ over MIP, comparing intrinsic information about the **maximal cause–effect state** only; requires both cause and effect power (min) |
| Mechanism partitions | Bipartitions of (mechanism, purview) into two pairs $\{M^{(1)}/Z^{(1)}, M^{(2)}/Z^{(2)}\}$ (parts possibly empty), factorized repertoires via "virtual elements" | Set $\Theta(M,Z)$ (Eq 38) of disjoint partitions into $k \ge 2$ (possibly empty) mechanism/purview part-pairs, with $M^{(i)} = M \Rightarrow Z^{(i)} = \emptyset$: always genuinely "disintegrating"; product probabilities formalized (no virtual-element language); MIP normalized by max pairwise interactions cut |
| Purview/state selection order | φ^max: purview maximizing EMD-based φ | State first by max ii per purview (Eq 36), then purview by max φ (Eq 45); φ_d = min(φ_c, φ_e) |
| Congruence requirement | None (concepts existed regardless of any system-level state) | Distinctions must have $z^{*}_c \subseteq s'_c$, $z^{*}_e \subseteq s'_e$ — congruent with the system's maximal cause–effect state; else excluded from the Φ-structure |
| Relations | Not in the formalism (similarity between concepts implicit in XEMD qualia-space geometry) | Explicit first-class objects: faces, overlaps, $\varphi_r$ (Eq 55), analytic $\sum \varphi_r$ and counts; up to $3^{|\mathbf d|}$ faces per relation |
| Big Phi (Φ) | Distance (XEMD) between the conceptual structure of the whole system and that of the MIP-partitioned system — only concepts *affected by the MIP* contribute; involves a partition | $\Phi = \sum_{C} \varphi = \sum \varphi_d + \sum \varphi_r$ over the full Φ-structure; **no partition, no normalization**; tracks "richness" of all contents |
| Terminology | Concepts; conceptual structure; qualia space; MICS | Distinctions; relations; cause–effect structure; **Φ-structure**; Φ-folds |
| Ties | Unaddressed (known non-uniqueness criticisms) | Operational resolution procedure (S1 Text), maximal-existence based; unresolved intrinsic ties disqualify complexes |
| Axioms | 5 axioms incl. existence folded into information | 0th axiom (existence) separated; 5 axioms: intrinsicality, information, integration, exclusion, composition (composition last); postulates track axioms more exactly |
| State-dependence | State-dependent (as 2.0) | State-dependent; requires **both** positive cause and effect power (as 1.0/3.0; 2.0 evaluated only causes) |
| Unit grains | Macro/micro analysis external (Hoel 2016) | Grain selection integrated via exclusion + maximal existence; units must be maximally irreducible within |

---

## 10. Glossary of symbols (as used in the 4.0 paper)

| Symbol | Meaning |
|---|---|
| $U$, $u$, $\Omega_U$ | Universe/substrate (set of units), its current state (set of unit–state tuples), state space |
| $\mathcal{T}_U$ | Complete interventional TPM $p(\bar u \mid u)$, Eq (1) |
| $do(u)$ | Pearl intervention setting state $u$ |
| $S$, $s$ | Candidate system $\subseteq U$ and its current state |
| $W$, $w$ | Background units $U \setminus S$ and their current state |
| $\mathcal{T}_e$, $\mathcal{T}_c$ | Effect / cause TPMs of $S$ with background causally marginalized conditional on $u$ (Eqs 3–4) |
| $\bar s$, $\hat s$ | A potential cause/effect state of $S$; a dummy summation state |
| $p_e(\bar s \mid s)$, $p_e(\bar s)$ | Constrained / unconstrained (uniform-average) effect probability |
| $p_c(s \mid \bar s)$, $p_c(s)$ | Forward cause-side probability / its uniform average |
| $p_c^{\leftarrow}(\bar s \mid s)$ | Backward cause probability (Bayes, uniform prior), Eq (9) |
| $ii_e$, $ii_c$ | Intrinsic effect/cause information (selectivity × informativeness), Eqs (5), (7); ibits |
| $s' = \{s'_c, s'_e\}$ | Maximal cause–effect state of the system (argmax of ii) |
| $\Theta(S)$, $\theta$, $\delta_i$ | Set of directional system partitions; a partition; per-part direction flag $\in \{\leftarrow, \rightarrow, \leftrightarrow\}$ |
| $S^{(i)}$, $X^{(i)}$, $Y^{(i)}$ | Partition part; units whose inputs to $S^{(i)}$ are cut; complement |
| $\mathcal{T}^{\theta}_{e/c}$, $p^{\theta}$ | Partitioned TPMs/probabilities (cut connections noised), Eqs (17–18) |
| $\varphi_c, \varphi_e$ | Cause/effect integrated information of system (Eqs 19–20) or mechanism (Eqs 41, 44) |
| $\varphi_s$ | System integrated information $\min(\varphi_c, \varphi_e)$ over the MIP ("small phi") |
| $\theta'$ | Minimum partition (MIP), normalized argmin (Eqs 23, 43) |
| $\varphi_s^{*}$ | Maximum $\varphi_s$ among candidate systems — defines the complex |
| $S^{*}$, $s^{*}$ | Maximal substrate (complex) and its state; "monad" = single-unit complex |
| $J$, $j$, $g$, $\hat S$ | Macro unit, its state, state mapping $g: \Omega_{\hat S} \to \Omega_J$, its micro substrate |
| $M$, $m$ | Mechanism $\subseteq S$ and its state $\subseteq s$ |
| $Z$, $z$; $Z_c$, $Z_e$ | Purview and purview state; cause/effect purviews |
| $X = S \setminus M$, $Y = S \setminus Z$ | Non-mechanism / non-purview system units (uniformly causally marginalized) |
| $\pi_e(z \mid m)$, $\pi_c(m \mid z)$ | Product effect/cause probabilities (Eqs 29–30) |
| $\pi_e(z; M)$, $\pi_c(m; Z)$ | Unconstrained product probabilities (Eqs 31–32) |
| $\pi_c^{\leftarrow}(z \mid m)$ | Backward product cause probability (Eq 33) |
| $z'_{c/e}(m, Z)$ | Maximal cause/effect state of $m$ within purview $Z$ (argmax ii) |
| $\Theta(M, Z)$ | Disintegrating mechanism partitions (Eq 38); $\mathcal{P}$ = power set |
| $z^{*}_{c/e}(m)$ | Maximally irreducible cause/effect (purview + state) of $m$ (argmax φ over purviews) |
| $\varphi_d$ | Distinction integrated information $\min(\varphi_c(m), \varphi_e(m))$ |
| $d(m) = (m, z^{*}, \varphi_d)$ | Causal distinction |
| $D(\mathcal{T}_e, \mathcal{T}_c, s)$ | Set of congruent distinctions with $\varphi_d > 0$ (Eq 48) |
| $\mathbf{d}$, $\mathbf{z}$ | A set of distinctions; a relating set of cause/effect purviews |
| $o^{*}(\mathbf{z})$, $o^{*}_f$ | Maximal (face) overlap = face purview (Eq 50) |
| $f(\mathbf{z}) = (\mathbf{z}, o^{*})$, $\mathbf{f}(\mathbf{d})$ | Relation face; set of faces of a relation |
| $r(\mathbf{d}) = (\mathbf{d}, \mathbf{f}, \varphi_r)$ | Causal relation; degree = $|\mathbf{d}|$; self-relation for $|\mathbf{d}| = 1$ |
| $\varphi_r$ | Relation integrated information (Eq 55) |
| $R(D)$ | Set of relations with $\varphi_r > 0$ (Eq 56) |
| $C(D) = D \cup R(D)$ | Cause–effect structure; for a complex: **Φ-structure** |
| $\Phi$ | Structure integrated information ("big phi") $= \sum_{C} \varphi$ (Eq 59) |
| Φ-fold | Sub-structure of a Φ-structure (distinction / compound / content Φ-folds) |
| ibit | Unit of ii/φ/Φ: point-wise bit weighted by a probability |
| MIP | Minimum partition; MICS (3.0 term) not used in 4.0 |
| $|\cdot|_+$ | Positive part, $\max(\cdot, 0)$ |

---

## 11. Ambiguities / underspecified points (defined only by the PyPhi `iit-4.0` branch or left open)

1. **Mechanism-partition normalization constant.** Eq (43) says the denominator is
   "the number of possible pairwise interactions affected by the partition" but never
   gives the closed form for $\Theta(M,Z)$ partitions (the system-level analogue
   $\sum_i |S^{(i)}||X^{(i)}|$ is given). The exact count for mechanism partitions
   (empty parts, the $\emptyset$-purview convention) must be taken from Barbosa et al.
   2021 / PyPhi.
2. **Enumeration of Θ(S) and Θ(M,Z) up to equivalence.** Different labeled
   (partition, direction-flag) combinations can induce identical noised TPMs (identical
   $\{X^{(i)}\}$); e.g. certain flag assignments duplicate cuts. The paper does not
   specify deduplication; implementations must decide whether to canonicalize (affects
   only cost, not results, but affects tie counts among MIPs).
3. **Ties in practice.** The S1 Text procedure is normative but computationally
   explosive as stated (resolving purview-state ties "at the level of the cause–effect
   structure by maximizing Φ" would require recomputing Φ per tied option; the text
   itself falls back to heuristics: "select the state congruent with $s'$" within a
   purview, "typically favors larger purviews" across purviews). What PyPhi actually
   implements (deterministic orderings, tolerance for float ties) is the de facto
   specification.
4. **Numerical equality.** All ties/argmaxes are defined over real-valued quantities;
   the paper is silent on tolerances. PyPhi uses an epsilon (`PRECISION` config). Any
   reimplementation must pin an equality tolerance to reproduce results.
5. **Degenerate denominators.** Eq (4): if the current universe state $u$ is unreachable
   under every $(\hat u)$, $\sum_{\hat u} p(u \mid \hat u) = 0$ and $p(\bar w \mid u)$
   is undefined (this motivated the 4.0 background treatment, but zero-probability
   corner cases within it — e.g. $p_c(s) = 0$ in Eq (9), or $q = 0$ states inside logs —
   rely on conventions $0 \log 0 = 0$ etc. that are implicit). ID's domain excludes
   $p > 0, q = 0$; the formalism does not state how to handle a constrained probability
   exceeding zero where the unconstrained/partitioned probability is exactly zero
   (informativeness $\to \infty$); deterministic systems can hit this.
6. **Purview candidacy.** Whether $Z$ ranges over all non-empty subsets only (empty
   purview would make ii/φ meaningless) is implicit; whether a mechanism may select a
   cause purview but no effect purview (then $\varphi_d = 0$ via the min) is handled by
   the min but not discussed for edge cases where argmax sets are empty.
7. **Relation faces in practice.** Since faces do not enter $\varphi_r$ beyond the union
   of face overlaps (which S3 proves equals $\bigcap_d (z^*_c \cup z^*_e)$), face
   enumeration is only needed if the *type* of a relation is wanted. The $|\mathbf z|>1$
   condition in Eq (49) vs. self-relations ($\mathbf z = \{z^*_c(d), z^*_e(d)\}$,
   $|\mathbf z| = 2$) is consistent, but whether a self-relation exists when
   $z^*_c(d) = z^*_e(d)$ exactly (as unit–state sets, then $|\mathbf z| = 1$) is not
   addressed.
8. **Macro-unit machinery.** The mapping $g$, the derivation of macro TPMs, and the
   "maximally irreducible within" criterion for units reference a paper "in prep."
   (footnote 11; since released as the macro-units framework). The main paper only
   sketches the iterative grain search; no algorithm is given.
9. **Recursive complex search with ties across non-overlapping systems** is only
   defined for overlapping ties; the order in which equal-φ_s* non-overlapping
   complexes are extracted is unspecified (doesn't change the final condensation but
   changes labels "first/second complex").
10. **"Cut one" and other approximations** are mentioned (PyPhi docs) but not part of
    the theory; any approximate mode in `iitx` should be clearly flagged as such.
11. **State-conditional background for causes**: Eq (4) computes a per-system-unit sum
    over $\bar w$ with weight $p(\bar w \mid u)$ — the weight uses the *whole universe*
    current state $u$; it is a modeling choice (stated, but with no uniqueness claim)
    and differs from all prior versions; sensitivity of results to this choice is
    unexplored in the paper.
12. **Non-uniform ID chance distributions.** In IIT usage the reference distribution is
    always the uniform-prior interventional marginal; the Barbosa paper allows arbitrary
    $Q$. The uniqueness theorem is agnostic, but the 4.0 pipeline always uses uniform
    interventional priors — a library should keep the reference distribution explicit.

---

## 12. Shape of the computation (implementation-oriented recap)

```
TPM 𝒯_U (|Ω_U| × |Ω_U|, factorized per-unit: n tensors p(ū_i | u))
 └─ for each candidate system S ⊆ U (2^n − 1 of them):
     ├─ condition/marginalize background → 𝒯_e, 𝒯_c            [tensor ops]
     ├─ ii_c/ii_e over all states \bar s → argmax → s' = {s'_c, s'_e}   [dense + argmax]
     ├─ for each directional partition θ ∈ Θ(S):                 [combinatorial]
     │    noised TPMs → φ_c, φ_e → φ_s(θ)/norm(θ)
     ├─ min over θ → MIP → φ_s(S)
 └─ argmax over S (recursively, removing winners) → complexes S*
     └─ for the complex: for each mechanism m ⊆ s (2^|S| − 1):
         ├─ for each purview Z ⊆ S (2^|S| − 1):
         │    ├─ π-probabilities, ii(m,z) over z ∈ Ω_Z → z'
         │    └─ min over θ ∈ Θ(M,Z) (normalized) → φ(m,Z)
         ├─ argmax over Z → z*_c, z*_e; φ_d = min(φ_c, φ_e)
         └─ congruence filter (z* ⊆ s') → D
     └─ relations: analytic Σφ_r and count from {(z*_c ∪ z*_e, φ_d)}   [sort/scan]
     └─ Φ = Σ φ_d + Σ φ_r
```

Combinatorial magnitudes: candidate systems $2^n$; system partitions ~ (ordered set
partitions × direction flags); mechanisms × purviews $\approx 4^n$; mechanism partitions
grow super-exponentially in $|M| + |Z|$; relations $\le 2^{2^n - 1} - 1$ but never
enumerated (analytic). Everything except the argmax/min/tie logic is dense linear
algebra over probability tensors and is trivially batchable.
