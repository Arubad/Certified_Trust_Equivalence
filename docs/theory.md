# Formal guarantees

This note states and proves the guarantees behind `tec/certify.py` and `tec/monitor.py`.
Notation follows REPORT.md §3. Four results are numbered: Theorem 1 (selective
certificate), Proposition 2 (audit-martingale delay), Corollary 3 (group certificates) and
Proposition 4 (predictable adaptive auditing). Each one ends with a **Checked by** line
that names the test or experiment that checks it numerically.

## 0. Setting and notation

An input $x$ has compressed-model score $s = s(x) \in [0,1]$ (top-1 minus top-2 probability
of C) and disagreement indicator $d = \mathbf 1\{C(x) \neq L(x)\}$. $P$ is the joint law of
$(s, d)$ under the calibration distribution. We write

$$\eta(u) = P(d = 1 \mid s = u), \qquad R(\lambda) = P(d = 1 \mid s \ge \lambda), \qquad \varphi(\lambda) = P(s \ge \lambda).$$

C serves $x$ iff $s(x) \ge \lambda$. The calibration sample is $(s_i, d_i)_{i=1}^n$, drawn
i.i.d. from $P$ (**Assumption I**). For a threshold $\lambda$, $n(\lambda) = \#\{i : s_i \ge \lambda\}$
and $k(\lambda) = \#\{i : s_i \ge \lambda,\ d_i = 1\}$. The p-value for
$H_0(\lambda): R(\lambda) > \alpha$ is

$$p(\lambda) = F_{n(\lambda),\alpha}\big(k(\lambda)\big), \qquad F_{m,a}(k) = P(\mathrm{Bin}(m,a) \le k)$$

(`binom_pvalue_upper`, with $p = 1$ if $n(\lambda) = 0$). Let
$m_0 = \max(20, \lceil \log\delta / \log(1-\alpha) \rceil)$ (`min_accept`).

**Procedure (`SelectiveCertificate.fit`).** Take a grid $\lambda_1 > \lambda_2 > \dots > \lambda_G$.
Walk it in order and skip every $\lambda_j$ with $n(\lambda_j) < m_0$. Among the rest, if
$p(\lambda_j) \le \delta$, set $\hat\lambda \leftarrow \lambda_j$ and continue; otherwise
stop. Initially $\hat\lambda = +\infty$ (serve nothing), and $R(+\infty) := 0$. In the code,
the grid is `unique(quantile(s_cal, linspace(0,1,G)))[::-1]`, so **it is computed from the
same calibration scores that are tested.**

Two facts are used repeatedly.

**(F1) Super-uniformity of a binomial CDF.** If $K \sim \mathrm{Bin}(m, \rho)$ with $\rho \ge \alpha$, then
$P(F_{m,\alpha}(K) \le u) \le u$ for all $u \in [0,1]$. More generally this holds for any $K$
that is stochastically larger than $\mathrm{Bin}(m,\alpha)$.
*Proof.* $F_{m,\alpha}$ is nondecreasing, so for $K' \sim \mathrm{Bin}(m,\alpha)$ and $K \succeq K'$ we have
$P(F_{m,\alpha}(K) \le u) \le P(F_{m,\alpha}(K') \le u)$. For a discrete variable, $P(F(K') \le u) \le u$
holds for its own CDF $F$: let $k^\star$ be the largest atom with $F(k^\star) \le u$; then
the probability is $F(k^\star) \le u$. A sum of independent Bernoulli($\rho_i$) with all
$\rho_i \ge \alpha$ is stochastically larger than $\mathrm{Bin}(m,\alpha)$ (coordinate-wise coupling). $\square$

**(F2) Fixed-sequence argument.** Let hypotheses be tested in a data-independent order and
suppose the procedure stops at the first non-rejection. Then every rejected hypothesis was
preceded only by rejections. So if any true null is rejected, the *first* true null
(in that order) is rejected.

## 1. Theorem 1 (selective certificate)

**Theorem 1.** Assume I.

**(a) Fixed grid.** Suppose the grid $\lambda_1 > \dots > \lambda_G$ and a start index $j_0$
are fixed independently of the calibration sample (for example, a lattice on $[0,1]$, or
quantiles of a held-out split). Run the procedure from $j_0$ with no skip rule. Then

$$P\big(R(\hat\lambda) \le \alpha\big) \ge 1 - \delta.$$

**(b) The implemented procedure (data-dependent quantile grid with skip).** Additionally
assume:

* (C) $s$ has a continuous distribution (no ties), and
* (M) a single crossing. Let $c = \inf\{t : R(t) \le \alpha\}$. Assume $R$ is continuous,
  $R(t) \le \alpha$ for **all** $t \ge c$, and $\eta(u) \ge \alpha$ for almost every $u < c$.

  The middle condition is needed. Without it, $R$ can rise above $\alpha$ again at higher
  thresholds. For example, take $s \sim U(0,1)$ and $\eta = 1$ on $[.99, 1]$, $0$ on
  $[.5, .99)$, $0.2$ on $[0, .5)$. Then $c = 0.4$ and the $\eta$-condition holds, but
  $R(t) > \alpha$ for all $t \in (0.8, 1]$.

  (M) holds in particular when $\eta$ is nonincreasing, that is, when a higher margin
  never means more disagreement. Then $R$ is nonincreasing, so $R(t) \le \alpha$ for
  $t \ge c$. Also, if $\eta(u_0) < \alpha$ for some $u_0 < c$, then $\eta < \alpha$ on
  $[u_0, 1]$, so $R(u_0) < \alpha$, which contradicts $u_0 < c$.

Under (C) and (M), the conclusion of (a) holds for the code as written. It also holds for
any grid, data-dependent or not, for which every count $n(\lambda_j)$ is a deterministic
function of $j$, $n$ and $G$, and for any fixed grid with the skip rule.

**(c) Minimum calibration size.** If $n < \lceil \log\delta / \log(1-\alpha) \rceil$, no
$\lambda$ can be certified: even with $k = 0$, $p = (1-\alpha)^{n(\lambda)} \ge (1-\alpha)^n > \delta$.
So $n \ge \lceil \log\delta/\log(1-\alpha)\rceil$ is necessary. For $\alpha = 0.05$ and
$\delta = 0.1$ the bound is 45. Because the code uses $m_0 \ge 20$, it needs
$n \ge m_0$ in practice. Neither condition is sufficient: a $\lambda$ also needs
$k(\lambda)$ small.

*Proof of (a).* Fix $j$. Conditional on the index set $A_j = \{i : s_i \ge \lambda_j\}$, the
pairs indexed by $A_j$ are i.i.d. draws from $P(\cdot \mid s \ge \lambda_j)$, because the
pairs are i.i.d. and membership depends only on each pair's own score. So
$k(\lambda_j) \mid n(\lambda_j) \sim \mathrm{Bin}(n(\lambda_j), R(\lambda_j))$. If $H_0(\lambda_j)$
is true, then $R(\lambda_j) > \alpha$ and (F1) gives $P(p(\lambda_j) \le \delta \mid n(\lambda_j)) \le \delta$,
hence also unconditionally. The error event $\{R(\hat\lambda) > \alpha\}$ means that
$\hat\lambda$ is a rejected true null. With a fixed order, (F2) says it implies rejection of the
fixed first true null $j^\star$. So $P(\text{error}) \le P(p(\lambda_{j^\star}) \le \delta) \le \delta$.
If there is no true null, the error probability is 0. $\square$

*Proof of (b).* By (M), the true nulls are exactly the thresholds $t < c$, where $R(t) > \alpha$:
by the definition of $c$ (an infimum), $R > \alpha$ below $c$, and $R \le \alpha$ on $[c, \infty)$
by assumption. If $c$ is at or below the lower edge of the support of $s$, there are no
true nulls and the error probability is 0. So assume $c$ lies above that edge.
Let $N = \#\{i : s_i \ge c\}$. Let $\mathcal G$ be the $\sigma$-field generated by $N$ and by the
scores of the points with $s_i < c$. Conditional on $\mathcal G$:

1. The $N$ points above $c$ are i.i.d. from $P(\cdot \mid s \ge c)$, whose disagreement rate
   is $R(c) = \alpha$. This follows from continuity of $R$, because $R > \alpha$ just below $c$
   and $R(c) \le \alpha$. So their disagreement count is
   $K_+ \sim \mathrm{Bin}(N, \alpha)$. If $P(s \ge c) = 0$, then $N = 0$ a.s. and $K_+ = 0$.
2. The points below $c$ have known scores, and their $d_i$ are independent
   $\mathrm{Bern}(\eta(s_i))$ with $\eta(s_i) \ge \alpha$ by (M). They are also independent of $K_+$.

A threshold $\lambda_j < c$ accepts all $N$ points above $c$ plus some below-$c$ points. So
$n(\lambda_j) \ge N$, and $k(\lambda_j) = K_+ + (\text{a sum of below-}c\text{ Bernoullis})$
is stochastically larger than $\mathrm{Bin}(n(\lambda_j), \alpha)$.

Now pick an index $J'$ that every error must pass through:

* In the code's quantile grid (no ties), $n(\lambda_j) = m_j$ is a deterministic ladder,
  and the active set $\{j : m_j \ge m_0\}$ is deterministic. Any $\lambda_j < c$ has
  $m_j \ge N$. Let $J'$ be the first active index with $m_{J'} \ge N$; it is a function of
  $N$ alone. Because the active order is deterministic and the walk stops at the first
  non-rejection, rejecting any true null forces a rejection at $J'$. This holds whether
  $\lambda_{J'}$ itself is above or below $c$ (linear interpolation allows both when
  $m_{J'} = N$). The set accepted at $J'$ is the top $m_{J'}$ points: all $N$ points above
  $c$ plus the top $m_{J'} - N$ points below $c$, which are $\mathcal G$-measurable.
* For a fixed grid with the skip rule, the true nulls are the $j$ with $\lambda_j < c$,
  and their counts $n(\lambda_j) = N + \#\{i : c > s_i \ge \lambda_j\}$ are $\mathcal G$-measurable.
  So $J' = $ the first $j$ with $\lambda_j < c$ and $n(\lambda_j) \ge m_0$ is $\mathcal G$-measurable.

In both cases, error implies rejection at $J'$, and $J'$ and its accepted set are
$\mathcal G$-measurable. By 1–2 and (F1), $P(\text{error} \mid \mathcal G) \le P(p(\lambda_{J'}) \le \delta \mid \mathcal G) \le \delta$. $\square$

**Is the data-dependent grid harmless? An honest assessment.** Proof (a) uses the fact
that the hypotheses and their order are fixed before the data are seen. The code's grid
breaks that: $\lambda_j$ is an order statistic of the scores being tested, so "is $H_0(\lambda_j)$
true?" is a random event correlated with the test statistic. The skip rule has the same
problem on a fixed grid, because whether $\lambda_j$ is active depends on $n(\lambda_j)$.

Each *single* test stays valid conditionally: given $\lambda_j = t$, the points strictly
above are i.i.d. from $P(\cdot \mid s > t)$. But (F2) needs the *first* true null to be
valid conditional on being first, and "being first" depends on how many points fell in the
upper region. Without (M), conditioning on that can favour low-risk compositions. We found
no proof of validity without (M), and none of the standard LTT/RCPS results cover it.

We did not find a counterexample either. A quick simulation with the actual class
($\alpha = 0.05$, $\delta = 0.1$, $n \in \{200, 1000\}$, 3000 reps) on non-monotone
$\eta$ — oscillating around $\alpha$, a high-risk band below a clean top, and a step
straddling $\alpha$ — gave violation rates of 0.3–5.3 %, all below $\delta$. (M) is plausible
for a margin score but it is an assumption. With ties ($s$ discrete, as for a heavily
quantised C), (C) fails, the counts stop being deterministic, and only part (a) applies.

**Recommendation (minimal fix, no code changed here).** For an assumption-free guarantee:

* build the grid from data that is not tested (a held-out split of the calibration set,
  or a fixed lattice such as `linspace(1, 0, G)` on the margin), and
* replace the data-dependent `continue` (skip) with a start index $j_0$ chosen from that
  held-out split. For example, take the first $\lambda_j$ whose held-out coverage times $n$
  is at least $2m_0$. Then do not skip: a test with $n(\lambda_{j_0}) < m_0$ simply fails
  and returns $+\infty$, which is conservative.

This is Theorem 1(a) verbatim. It costs a little power (split) or resolution (lattice).

**Checked by:** `tests/test_certify.py::test_certificate_validity_synthetic`, and the empirical violation
rate $\le \delta$ in `results/real/exp6a_summary.csv`.

## 2. Proposition 2 (audit-martingale detection delay)

**Monitor (`ShadowAuditMartingale`).** The monitor sees only served inputs. Each served
input is audited independently with probability $b$, and the $n$-th audit reveals $d_{(n)}$.
For $q \in Q$ (default $Q = \{\min(.95, 2\alpha), \min(.95, 4\alpha)\}$),

$$\log M^q_n = \sum_{i \le n} X^q_i, \qquad X^q_i = d_{(i)}\log\tfrac{q}{\alpha} + (1-d_{(i)})\log\tfrac{1-q}{1-\alpha},$$

$M_n = |Q|^{-1}\sum_q M^q_n$, and the alarm fires at $N_\delta = \inf\{n : M_n \ge 1/\delta\}$ audits.

**Post-change model.** After the change, stream inputs are i.i.d. A fraction $c$ is served
(coverage), and served inputs have disagreement rate $r$, i.e.
$d_{(i)} \overset{iid}\sim \mathrm{Bern}(r)$. The audit coins are independent of everything
else. Define the per-audit log-drift

$$\mathrm{KL}_q(r) = \mathbb E_r[X^q] = r\log\tfrac{q}{\alpha} + (1-r)\log\tfrac{1-q}{1-\alpha} = \mathrm{KL}(r\|\alpha) - \mathrm{KL}(r\|q),$$

and $\kappa(r) = \max_{q \in Q}\mathrm{KL}_q(r)$, attained at $q^\star$.

**Proposition 2.** Let $\tau$ be the detection delay counted in stream inputs (served and
deferred), with the monitor started at the change point.

1. If $\kappa(r) \le 0$, every component has nonpositive drift and $E[\tau] = \infty$.
   This always holds for $r \le \alpha$, where $\mathrm{KL}_q(r) < 0$ for all $q > \alpha$.
   It also holds on a **blind zone** $\alpha < r \le r_{\min}$, with
   $r_q = \log\frac{1-\alpha}{1-q} \big/ \log\frac{q(1-\alpha)}{\alpha(1-q)}$ and
   $r_{\min} = \min_q r_q$. For $\alpha = .05$: $r_{2\alpha} = 0.072$ and $r_{4\alpha} = 0.110$.
   For $r < r_{\min}$ the alarm probability is $< 1$.
2. If $\kappa(r) > 0$,

$$E[\tau] = \frac{E[N_\delta]}{b\,c}, \qquad \frac{\log(1/\delta)}{\kappa(r)}\,(1+o(1)) \;\le\; E[N_\delta] \;\le\; \frac{\log(|Q|/\delta) + \log(q^\star/\alpha)}{\kappa(r)},$$

where $o(1) \to 0$ as $\delta \to 0$. We use the computable approximation

$$\boxed{\;D(r) = \frac{\log(1/\delta) + \log|Q|}{b \cdot c \cdot \max_{q\in Q}\mathrm{KL}_q(r)}\;}\quad (D = \infty \text{ if } \max_q \mathrm{KL}_q(r) \le 0).$$

$D$ depends only on (`r, alpha, delta, budget, coverage, qs`), with `qs` defaulting to $Q$
above. This is `expected_audit_delay`.

*Proof.* Inputs to audits: each stream input is audited with probability $bc$, independently
(served, then coin). So the gaps $G_1, G_2, \dots$ between audits are i.i.d.
$\mathrm{Geom}(bc)$, independent of the $d_{(i)}$. $N_\delta$ is a stopping time of the audit
sequence, and $\tau = \sum_{i \le N_\delta} G_i$. Wald's identity gives
$E\tau = E[N_\delta]/(bc)$ whenever $E N_\delta < \infty$.

Upper bound: because $M_n \ge M^{q^\star}_n / |Q|$, we have
$N_\delta \le N^\star := \inf\{n : \log M^{q^\star}_n \ge \log(|Q|/\delta)\}$. $\log M^{q^\star}$ is a
random walk with i.i.d. bounded increments of mean $\kappa > 0$, so $E N^\star < \infty$.
Wald gives $\kappa\,E N^\star = \log(|Q|/\delta) + E[\text{overshoot}]$. The overshoot is
less than the largest positive increment, $\log(q^\star/\alpha)$.

Lower bound: $M_n \le \max_q M^q_n$, so the alarm needs some $\log M^q_n \ge \log(1/\delta)$.
By the SLLN, $\log M^q_n/n \to \mathrm{KL}_q(r) \le \kappa$ a.s. So the first passage of
$\max_q \log M^q$ over $h = \log(1/\delta)$ satisfies $N/h \to 1/\kappa$ a.s. as $h \to \infty$.
Fatou's lemma gives $\liminf_h E[N]/h \ge 1/\kappa$. Because $N \le N^\star$ and
$N^\star/h$ is uniformly integrable (renewal theory), the limit holds in mean (the
standard first-order SPRT asymptotics; Lorden 1970). Part 1: if $\kappa < 0$, every
$\log M^q$ has negative drift, so with positive probability all of them stay below
$\log(1/\delta)$ forever, the alarm probability is $< 1$, and $E\tau = \infty$. At
$\kappa = 0$ (only at $r = r_q$), the walk is recurrent: it alarms a.s. but with
infinite mean delay. $\square$

**Is the band $[0.7, 1.6] \times D$ justified?** $D$ sits between the two Wald terms. The
corrections are:

* (i) *Overshoot*: at most $+\log(q^\star/\alpha)/\log(|Q|/\delta)$ relative. For $|Q| = 2$
  and $q^\star = 4\alpha$: $\log 4/\log 40 = 0.38$ at $\delta = .05$, and
  $\log 4/\log 20 = 0.46$ at $\delta = .1$. So the ratio is at most 1.46.
* (ii) *Mixture* $\log|Q|$: the other component also contributes, so the true threshold
  lies between $\log(1/\delta)$ and $\log(|Q|/\delta)$. The ratio is about
  $\log(1/\delta)/\log(|Q|/\delta) = 0.81$ ($\delta = .05$) or $0.77$ ($\delta = .1$) at
  the low end. This is heuristic: the lower bound is only first-order asymptotic
  ($o(1)$ as $\delta \to 0$), and it is not a finite-$\delta$ bound. The
  effect is largest when the two $\mathrm{KL}_q$ are close (near $r \approx 0.12$–$0.15$ for
  $\alpha = .05$).
* (iii) *Discreteness/fluctuation*: with few audits to alarm (large $r$), the walk crosses
  in lattice steps. This adds overshoot and is bounded by (i).

Near the blind-zone edge, only $q^\star = 2\alpha$ has positive drift. The
negative-drift component drops out, so the effective threshold is
$\approx \log(|Q|/\delta)$, the mixture does not help, and the ratio is
$\approx 1 + \text{overshoot}/\log(|Q|/\delta) \le 1 + \log 2/\log(|Q|/\delta)$. That is
$\le 1.19$ at $\delta = .05$ and $\le 1.23$ at $\delta = .1$. The bracket is therefore
roughly $[0.77, 1.46]$ (heuristic at the low end), and $[0.7, 1.6]$ adds Monte-Carlo slack.

*Scope.* $D$ assumes the monitor starts at the change point. In exp4 and exp5, the monitor
runs before the change, and $\log M$ drifts negative during that time (there is no floor).
So the delays measured from the change exceed $D$; $D$ is not a prediction for those
experiments.

A check of $E\tau/D$ (800–3000 reps, $b = .1$, $c \in \{.8, 1\}$, $\alpha \in \{.05, .1\}$,
$\delta \in \{.01, .05, .1\}$, $r$ from $0.075$ to $0.5$) gave ratios in $[0.83, 1.12]$.
Keep $r$ outside the blind zone by a margin, because $\mathrm{Var}(\tau)$ blows up as
$\kappa \to 0$, and use $\ge 500$ reps.

**Checked by:** `tests/test_theory.py::test_delay_matches_simulation` (ratio in
$[0.7, 1.6]$) and `tests/test_theory.py::test_delay_inf_under_null`: $D = \infty$ for $r \le \alpha$ and for
a blind-zone rate ($r = 0.06$, $\alpha = .05$).

### 2.1 False-alarm guarantee, its $H_0$, and resets

The audited process is indexed by audits. Its filtration $\mathcal F_{n-1}$ is generated by
the first $n-1$ audited outcomes, and by the coins and scores of inputs *strictly before*
the $n$-th audited input. The $n$-th audited input's own score is **not** in
$\mathcal F_{n-1}$. If it were, the conditional rate would be $\eta(s)$, not
$R(\hat\lambda)$, and the null below could fail. The $H_0$ is
**conditional on being served**:

$$H_0:\quad P(d_{(n)} = 1 \mid \mathcal F_{n-1}) \le \alpha \quad\text{for all } n.$$

For $q > \alpha$, $E[e^{X^q_n} \mid \mathcal F_{n-1}] = 1 + (\rho_n - \alpha)\frac{q-\alpha}{\alpha(1-\alpha)} \le 1$,
where $\rho_n$ is the conditional rate. So each $M^q$, and the average $M$, is a nonnegative
supermartingale with $M_0 = 1$. Ville's inequality gives $P(\exists n : M_n \ge 1/\delta) \le \delta$
over the whole stream.

A sufficient condition is that served inputs are i.i.d. over time
(stationary $P$, fixed $\hat\lambda$) with $R(\hat\lambda) \le \alpha$. Then $\rho_n = R(\hat\lambda)$
for every $n$. The null also allows non-stationary streams whose served risk stays
$\le \alpha$ at every step.

Two caveats:

* The certificate itself fails with probability $\le \delta_{\text{cert}}$. Without drift,
  $P(\text{alarm}) \le \delta_{\text{cert}} + \delta$, and an alarm caused by a failed
  certificate is a true detection.
* **Reset after recalibration.** `reset()` starts a fresh $M$ with $M_0 = 1$, and
  recalibration selects a new $\hat\lambda_k$ on 300 L-labelled stream inputs that are not
  used again by the monitor. Given $\hat\lambda_k$ and the past, epoch $k$ is a new Ville test
  with the same $H_0$ for $\hat\lambda_k$. Over $K$ epochs, the probability of at least one
  false alarm is $\le K\delta$ (union bound). The guarantee is per epoch, not per stream.
  For the new certificate to be valid, the 300 inputs must be i.i.d. from the distribution
  that later stream inputs follow. If drift is continuing, that assumption fails.

## 3. Corollary 3 (group certificates)

**Setting (`GroupCertificate`).** The group $g \in \mathcal G$ is **observed** for every
calibration point and at serving time (**Assumption O**). Each group gets its own
$\hat\lambda_g$ from Theorem 1 at level $\delta_g$. Group risk is
$R_g(\lambda) = P(d = 1 \mid s \ge \lambda, G = g)$.

**Corollary 3.** Under I and O, and the conditions of Theorem 1(a), or (b) per group:

1. For each $g$: $P(R_g(\hat\lambda_g) \le \alpha) \ge 1 - \delta_g$. This is a per-group guarantee.
2. Simultaneously: $P(\forall g : R_g(\hat\lambda_g) \le \alpha) \ge \prod_g (1-\delta_g) \ge 1 - \sum_g \delta_g$.

The code passes the same $\delta$ to every group, which guarantees each group at
$1-\delta$. Jointly, coverage is only guaranteed to be at least
$(1-\delta)^{|\mathcal G|} \approx 1 - |\mathcal G|\delta$, a lower bound. Equivalently, the
probability that some group fails is at most $1 - (1-\delta)^{|\mathcal G|}$.
Simultaneous coverage $1-\delta$ needs $\delta_g = \delta/|\mathcal G|$ (Bonferroni), or the
slightly larger $\delta_g = 1 - (1-\delta)^{1/|\mathcal G|}$ (Šidák, exact here).

*Proof.* Condition on the group labels $(g_i)$. Within group $g$, the $n_g$ pairs are i.i.d.
from $P(\cdot \mid G = g)$, and different groups' samples are independent. Theorem 1
applied to $P(\cdot \mid G = g)$, conditional on $n_g$, gives
$P(R_g(\hat\lambda_g) > \alpha \mid (g_i)) \le \delta_g$. A group with $n_g < m_0$ gets
$\hat\lambda_g = \infty$ and is trivially safe. The events for different groups are
conditionally independent, which gives the product. The inequality
$\prod_g(1-\delta_g) \ge 1 - \sum_g\delta_g$ (Weierstrass) gives the sum.
Integrate out $(g_i)$. $\square$

The independence step is what makes the product form valid. The union bound
$1 - \sum\delta_g$ needs no independence, so it also covers overlapping or
intersectional groups.

If Assumption O fails and $g$ is replaced by a predicted group $\hat g$, the guarantee
holds for $\hat g$-groups, not for true groups.

**Checked by:** `experiments/exp8_theory_checks.py` part (c): the empirical rate of
"some group violates" against $1 - (1-\delta)^{|\mathcal G|}$ with per-group $\delta$, and
$\le \delta$ with $\delta/|\mathcal G|$.

## 4. Proposition 4 (predictable adaptive auditing)

**Setting.** The monitor runs on stream index $t$. Let $\sigma_t = \mathbf 1\{s_t \ge \hat\lambda\}$
(served) and $A_t \in \{0,1\}$ (audit coin). Let
$\mathcal F_t = \sigma\big(s_{1:t}, \sigma_{1:t}, A_{1:t}, (A_i\sigma_i d_i)_{i \le t}, U\big)$,
where $U$ is any external randomness. The implemented rule (Task 3) is:

1. the rate $b_t \in [0,1]$ is computed from the KS window over $s_{1:t-1}$ and other
   $\mathcal F_{t-1}$ data;
2. $A_t \sim \mathrm{Bern}(b_t)$ is drawn with fresh randomness;
3. only then is $s_t$ appended to the window.

So $b_t$ is $\mathcal F_{t-1}$-measurable, and $A_t \perp (s_t, d_t) \mid \mathcal F_{t-1}$
(**Assumption P**). The audited martingale is

$$M^q_t = \prod_{i \le t} \Big[\big(\tfrac{q}{\alpha}\big)^{d_i}\big(\tfrac{1-q}{1-\alpha}\big)^{1-d_i}\Big]^{A_i\sigma_i}, \qquad M_t = |Q|^{-1}\textstyle\sum_q M^q_t.$$

**$H_0$:** $P(d_t = 1 \mid \mathcal F_{t-1}, \sigma_t = 1) \le \alpha$ for all $t$ with
$P(\sigma_t = 1 \mid \mathcal F_{t-1}) > 0$. This holds when $(s_t, d_t)$ are i.i.d. and
independent of the past, with $R(\hat\lambda) \le \alpha$.

**Proposition 4.** Under P and $H_0$, $M_t$ is a nonnegative $\mathcal F_t$-supermartingale
with $M_0 = 1$. Hence $P(\exists t : M_t \ge 1/\delta) \le \delta$, for any predictable
schedule $(b_t)$, including the KS-triggered one, and with no horizon.

*Proof.* Let $L^q_t = (q/\alpha)^{d_t}((1-q)/(1-\alpha))^{1-d_t}$. Then

$$E[(L^q_t)^{A_t\sigma_t} \mid \mathcal F_{t-1}] = 1 - b_t\,\pi_t + b_t\,\pi_t\,E[L^q_t \mid \mathcal F_{t-1}, \sigma_t = 1, A_t = 1],$$

where $\pi_t = P(\sigma_t = 1 \mid \mathcal F_{t-1})$. Here we used $P(A_t = 1 \mid \mathcal F_{t-1}, s_t) = b_t$
by P. Also by P, conditioning on $A_t = 1$ does not change the law of $d_t$ given
$(\mathcal F_{t-1}, \sigma_t = 1)$. So the inner expectation is
$1 + (\rho_t - \alpha)\frac{q-\alpha}{\alpha(1-\alpha)} \le 1$, with $\rho_t \le \alpha$ by $H_0$.
The whole expression is therefore $\le 1$, so $E[M^q_t \mid \mathcal F_{t-1}] \le M^q_{t-1}$.
Averages of supermartingales on a common filtration are supermartingales. Ville's
inequality completes the proof. $\square$

**Counter-example: score-dependent auditing $b_t = f(s_t)$ breaks it.** Let $\alpha = 0.05$
and $\hat\lambda = 0$. Let $s \sim U(0,1)$ with $\eta(s) = 0.1$ for $s < 1/2$ and $0$
otherwise. Then the served risk is $R = 0.05 \le \alpha$, so $H_0$ is true. Audit with
$f(s) = \mathbf 1\{s < 1/2\}$, i.e. "audit only the low-margin served inputs". The
audited $d$'s are i.i.d. $\mathrm{Bern}(0.1)$, so $\log M^{2\alpha}$ has drift
$\mathrm{KL}_{0.1}(0.1) = 0.1\log 2 + 0.9\log(0.9/0.95) = 0.0207 > 0$. By the SLLN,
$M_t \to \infty$ a.s., and the false-alarm probability is $1 > \delta$.

The step that fails is $A_t \perp d_t \mid \mathcal F_{t-1}$: the coin depends on $s_t$,
which carries information about $d_t$. This rules out any KS rule that uses $s_t$ itself
when deciding whether to audit $t$. That is why step 3 comes after step 2.

*Remark (valid score-dependent auditing).* If $b_t = f_t(s_t)$ with $f_t$ predictable and
$f_t \ge b_{\min} > 0$, the IPW e-process

$$E_t = \prod_{i \le t}\big(1 + \kappa_i\,\sigma_i\,(A_i d_i / f_i(s_i) - \alpha)\big),$$

with predictable $\kappa_i \in [0, 1/\alpha]$, is a nonnegative supermartingale under the
marginal $H_0$. This needs the coin to depend on the past and $s_i$ only:
$A_i \perp d_i \mid (s_i, \mathcal F_{i-1})$ with $P(A_i = 1 \mid s_i, \mathcal F_{i-1}) = f_i(s_i)$.
Then $E[A_i d_i / f_i(s_i) \mid s_i, \mathcal F_{i-1}] = \eta(s_i)$.

**Checked by:** `tests/test_monitor.py::test_adaptive_null_false_alarm` (KS-triggered
$b_t$ on an $H_0$ stream: false-alarm rate $\le \delta$ up to Monte-Carlo error).
