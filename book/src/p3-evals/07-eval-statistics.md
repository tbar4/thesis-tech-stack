# The statistics of evals

An eval score is an estimate, not a measurement. When the harness prints `accuracy: 0.72` it is handing you the mean of a sample, and a sample mean is a random variable with a distribution of its own. The entire discipline of this chapter is refusing to read that number as if it were exact. Every claim this book makes downstream (that a quantized model is as good as the full-precision one, that a training run moved the needle, that judge A beats judge B) is a claim about a difference between two such estimates, and a difference between two noisy estimates can be pure noise dressed up as a result. So this chapter builds the machinery that tells noise from signal: confidence intervals from the bootstrap, paired tests for the shared-item comparisons that dominate this book, corrections for running many tests at once, and a power analysis that tells me how many items I need before I bother running the comparison at all. The output is `evalstats`, a small module that every later comparison in the book imports. This is the chapter the thesis committee reads with a red pen, so I derive everything from definitions.

## Theory

### Where eval variance comes from

Before estimating uncertainty you have to know what is random. Five sources contribute, and they are not interchangeable.

The first and usually largest is **item sampling variance**: your test set is a finite sample of items from some larger population of tasks, and a different draw of items would give a different score. If accuracy is $p$ and you have $n$ independent items, the standard error of the estimated accuracy is on the order of $\sqrt{p(1-p)/n}$, which for $p=0.7$ and $n=200$ is about $\sqrt{0.21/200} \approx 0.032$, i.e. plus or minus three percentage points before you have done anything at all. That number alone kills a lot of over-confident benchmark comparisons.

The second is **decoding stochasticity**: with temperature above zero, the same model on the same item can be right one sample and wrong the next. The third is **judge or scorer noise**, which for a model-graded metric is the judge's own variance from Chapter 3.6. The fourth is **prompt and format variance**: few-shot example choice, ordering, and template wording move scores by amounts that rival real model differences. The fifth is **implementation variance**: tokenizer quirks, stop-sequence handling, and answer-extraction regexes, which is why Chapter 3.5 insisted on comparability.

The estimand (the thing you are trying to learn) is the population accuracy $p = \mathbb{E}[X]$ where $X$ is the per-item score. The estimator is the sample mean $\hat p = \frac1n\sum_i X_i$. Everything below is about putting honest error bars on $\hat p$ and on differences of two such $\hat p$'s.

### Seeds, resampling, and the two views of an item

There are two legitimate ways to treat an eval item, and confusing them is a common error. In the **fixed-item** view, the test set is *the* population you care about and the only randomness is decoding; here you reduce variance by fixing seeds and, if you want to characterize decoding noise, by drawing multiple completions per item and averaging. In the **random-item** view, the test set is a sample and you want to generalize to unseen items; here the item draw is the dominant randomness and the bootstrap resamples items. This book almost always takes the random-item view, because the point of an eval is to predict how the model does on tasks like these, not only on these exact tasks. When an item has multiple completions, the two sources stack, and the correct resampling unit is the *item* (a cluster of completions), not the individual completion, or you will understate the variance by pretending correlated completions are independent.

### The bootstrap

The bootstrap is how I get a confidence interval without assuming the score distribution is anything in particular. It is the workhorse of the whole module, so I derive it in full.

```admonish derivation
**The nonparametric bootstrap.** Let the per-item scores $X_1,\dots,X_n$ be i.i.d. draws from an unknown distribution $F$. We want a confidence interval for a statistic $\theta = T(F)$; take $\theta = \mathbb{E}_F[X]$, the population accuracy. The estimator is the plug-in $\hat\theta = T(F_n)$, where $F_n$ is the empirical distribution putting mass $1/n$ on each observed $X_i$; for the mean this is just $\hat\theta = \frac1n\sum_i X_i$.

The **plug-in principle** says: since $F_n \to F$, quantities computed from $F_n$ approximate the same quantities computed from $F$. In particular the sampling distribution of $\hat\theta - \theta$ (which we cannot compute, because we do not know $F$) is approximated by the distribution of $\hat\theta^* - \hat\theta$, where $\hat\theta^*$ is the statistic recomputed on a resample drawn from $F_n$. Drawing from $F_n$ is exactly sampling $n$ items *with replacement* from the observed data.

The procedure. For $b = 1,\dots,B$:
$$ X^{*(b)}_1,\dots,X^{*(b)}_n \overset{\text{iid}}{\sim} F_n \quad\text{(sample $n$ items with replacement)}, \tag{7.1}$$
$$ \hat\theta^{*(b)} = T\!\left(F_n^{*(b)}\right) = \frac1n\sum_{i=1}^n X^{*(b)}_i. \tag{7.2}$$
The collection $\{\hat\theta^{*(b)}\}_{b=1}^B$ is the **bootstrap distribution**. Two things fall out of it. The bootstrap standard error is its standard deviation,
$$ \widehat{\mathrm{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \left(\hat\theta^{*(b)} - \bar\theta^*\right)^2}, \qquad \bar\theta^* = \frac1B\sum_b \hat\theta^{*(b)}. \tag{7.3}$$
The **percentile interval** at level $1-\alpha$ takes the empirical quantiles of the bootstrap distribution,
$$ \left[\, \hat\theta^*_{(\alpha/2)},\ \hat\theta^*_{(1-\alpha/2)} \,\right], \tag{7.4}$$
where $\hat\theta^*_{(q)}$ is the $q$-th quantile of $\{\hat\theta^{*(b)}\}$. For a two-sided 95% interval, $\alpha = 0.05$ and you read off the 2.5th and 97.5th percentiles.

**Sanity check against the closed form.** For the mean of binary scores, $\hat\theta = \hat p$ and the classical standard error is $\sqrt{\hat p(1-\hat p)/n}$. The bootstrap SE in Eq. 7.3 converges to exactly this as $B\to\infty$, because resampling a Bernoulli sample reproduces its variance $\hat p(1-\hat p)$ and dividing the resample mean's variance by $n$ recovers the same formula. So the bootstrap does not contradict the textbook interval; it *generalizes* it to statistics (kappa, F1, a paired difference) that have no clean closed form.
```

Two refinements matter in practice. First, the **clustered (block) bootstrap**: when each item carries several completions, you resample *items* and take all completions of each resampled item together, which preserves within-item correlation and keeps the SE honest. Second, **BCa** (bias-corrected and accelerated) intervals adjust the percentile endpoints for skew and median-bias; they are strictly better than the plain percentile interval for skewed statistics, and the module exposes them, but for near-symmetric statistics like accuracy the percentile interval is fine and I default to it for transparency.

### Paired differences: A versus B on shared items

Almost every comparison in this book runs both models on the *same* items: full-precision versus quantized, before-training versus after, judge A versus judge B. Sharing items is a gift, because it lets you cancel item difficulty. The right analysis is paired, and paired analysis can be dramatically more powerful than treating the two runs as independent samples.

```admonish derivation
**The paired $t$-test and its assumptions.** For each shared item $i$, let $A_i$ and $B_i$ be the two models' scores and form the difference $d_i = A_i - B_i$. If the models are compared on the same items, the $d_i$ are i.i.d. draws with mean $\mu_d = \mathbb{E}[A - B]$, which is exactly the quantity we care about. The null hypothesis is $H_0: \mu_d = 0$.

Let $\bar d = \frac1n\sum_i d_i$ and $s_d^2 = \frac{1}{n-1}\sum_i (d_i - \bar d)^2$. The paired $t$ statistic is
$$ t = \frac{\bar d}{s_d/\sqrt{n}}, \tag{7.5}$$
which under $H_0$ follows a Student $t$ distribution with $n-1$ degrees of freedom. The power of pairing is visible in the denominator: $\mathrm{Var}(\bar d) = (\sigma_A^2 + \sigma_B^2 - 2\,\mathrm{Cov}(A,B))/n$. Because item difficulty makes $A$ and $B$ positively correlated, $\mathrm{Cov}(A,B) > 0$, so the paired variance is smaller than the unpaired $(\sigma_A^2 + \sigma_B^2)/n$. Pairing subtracts the shared difficulty out.

**Assumptions**, in the order they bite. (1) The $d_i$ are independent across items; violated if items share a passage or a template, which argues for a clustered bootstrap instead. (2) The $d_i$ are approximately normal, or $n$ is large enough for the CLT to carry $\bar d$; for continuous or averaged scores this is mild, but for *binary* per-item scores the differences live in $\{-1, 0, +1\}$ and normality is a stretch at small $n$. (3) No systematic pairing error. When assumption (2) is shaky, use the exact paired test for binary outcomes below, or the paired bootstrap, which assumes neither normality nor a variance formula.
```

For binary accuracy on shared items, the exact and correct paired test is **McNemar's test**, which throws away the items both models get right or both get wrong (they carry no information about the difference) and looks only at the disagreements.

```admonish derivation
**McNemar's test.** Cross-tabulate the shared items by (A correct?) $\times$ (B correct?):

| | B correct | B wrong |
|---|---|---|
| **A correct** | $a$ | $b$ |
| **A wrong** | $c$ | $d$ |

The concordant cells $a$ (both right) and $d$ (both wrong) tell you nothing about which model is better. All the signal is in the **discordant** cells: $b$ items where A is right and B is wrong, $c$ items where B is right and A is wrong. Under $H_0$ (the two models are equally likely to win a discordant item), each discordant item is a fair coin, so
$$ b \sim \mathrm{Binomial}(b + c,\ 1/2). \tag{7.6}$$
An exact two-sided p-value sums the binomial tail; the large-sample approximation is the chi-square statistic with one degree of freedom,
$$ \chi^2 = \frac{(b - c)^2}{b + c}, \qquad \text{or with continuity correction}\quad \chi^2_c = \frac{(|b - c| - 1)^2}{b + c}. \tag{7.7}$$
The estimated accuracy difference is $\hat\mu_d = (b - c)/n$, and its whole uncertainty comes from the $b + c$ discordant pairs, which is why the *discordant rate* $\psi = (b+c)/n$ is the quantity that drives the power analysis below.
```

The paired bootstrap is the assumption-light default I actually reach for: resample items with replacement, recompute $\bar d$ on each resample, and read the CI for the difference straight off Eq. 7.4. It handles binary or continuous scores, clustered completions, and weird statistics uniformly, which is why it anchors the module.

### Multiple comparisons

The moment you run more than one test, the probability of at least one false positive climbs. Compare a model against five baselines at $\alpha = 0.05$ each and, if all five nulls are true, the chance of at least one spurious "significant" result is $1 - (1-0.05)^5 \approx 0.23$. You have to correct, and there are two corrections worth knowing, controlling two different error rates.

```admonish derivation
**Bonferroni (controls the family-wise error rate).** Suppose you run $m$ tests and want the FWER, the probability of *any* false rejection among the true nulls, to be at most $\alpha$. Test each hypothesis at level $\alpha/m$. Let $R_i$ be the event that true null $i$ is falsely rejected; then $P(R_i) \le \alpha/m$. By **Boole's inequality** (the union bound),
$$ \mathrm{FWER} = P\!\left(\bigcup_{i} R_i\right) \le \sum_{i} P(R_i) \le m_0 \cdot \frac{\alpha}{m} \le \alpha, \tag{7.8}$$
where $m_0 \le m$ is the number of true nulls. No independence assumption is needed, which is Bonferroni's strength and, because it is conservative when tests are correlated, also its weakness.

**Benjamini-Hochberg (controls the false discovery rate).** When you have many tests and can tolerate a controlled *fraction* of false positives among your rejections rather than demanding almost none, control the FDR, $\mathbb{E}[V/R]$, the expected proportion of false rejections $V$ among all rejections $R$. Order the p-values $p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}$. Find
$$ k^* = \max\Big\{ k : p_{(k)} \le \tfrac{k}{m}\, q \Big\}, \tag{7.9}$$
and reject the hypotheses corresponding to $p_{(1)},\dots,p_{(k^*)}$. Benjamini and Hochberg proved that under independence (and under positive regression dependence, PRDS) this guarantees $\mathrm{FDR} \le \frac{m_0}{m} q \le q$. The intuition for the sloped threshold $\frac{k}{m}q$: the $k$-th smallest of $m$ uniform p-values has expectation $\approx k/(m+1)$ under the null, so comparing $p_{(k)}$ against $\frac{k}{m}q$ asks "is this p-value smaller than a null p-value at its rank would be, by a factor $q$?" BH is uniformly more powerful than Bonferroni and is the right default when you are screening many model or item comparisons rather than defending a single headline claim.
```

Which to use is a modeling decision, not a formula. For the one comparison that anchors a thesis chapter (does training help), you are not in multiple-comparison territory at all. For a sweep over many configs or per-stratum breakdowns, use BH. For a small pre-registered family where any single false positive would be embarrassing, use Bonferroni.

### Power: is my delta real, and how many items do I need

Power analysis inverts the question. Instead of "given my data, is the difference significant," it asks "given the difference I expect and the noise I have, how many items do I need for a real difference to clear significance with high probability." This is what sizes the thesis suite in Chapter 3.9, so I do it concretely.

```admonish derivation
**Sample size for a paired comparison.** Model the per-item difference $d_i$ as having true mean $\delta$ (the effect you hope to detect) and standard deviation $\sigma_d$. The two-sided test at level $\alpha$ rejects when $|t|$ exceeds $z_{1-\alpha/2}$; under the alternative, $\bar d$ is centered at $\delta$ with SE $\sigma_d/\sqrt n$, so power $1-\beta$ requires the alternative's center to sit $z_{1-\beta}$ standard errors past the critical value:
$$ \delta = \big(z_{1-\alpha/2} + z_{1-\beta}\big)\,\frac{\sigma_d}{\sqrt n} \;\Longrightarrow\; n \ge \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\delta}\right)^2. \tag{7.10}$$

**Specializing to paired accuracy (McNemar).** For binary shared-item scores, all the signal is in the discordant pairs. With discordant rate $\psi = p_{10} + p_{01}$ and effect $\delta = p_{10} - p_{01}$, a standard approximation for the required number of item pairs is
$$ n \approx \frac{\big(z_{1-\alpha/2}\sqrt{\psi} + z_{1-\beta}\sqrt{\psi - \delta^2}\big)^2}{\delta^2}. \tag{7.11}$$

**A worked number the suite will use.** Take $\alpha = 0.05$ two-sided ($z_{1-\alpha/2} = 1.96$), power $1-\beta = 0.80$ ($z_{1-\beta} = 0.8416$), and suppose a training run flips a quarter of the items in one direction or the other, $\psi = 0.25$. Then $\sqrt\psi = 0.5$. For a target detectable gain of $\delta = 0.08$ (eight accuracy points), $\sqrt{\psi - \delta^2} = \sqrt{0.25 - 0.0064} = 0.4936$, so
$$ n \approx \frac{(1.96\cdot 0.5 + 0.8416\cdot 0.4936)^2}{0.08^2} = \frac{(0.980 + 0.4154)^2}{0.0064} = \frac{1.947}{0.0064} \approx 304. $$
So roughly **300 shared items** detect an eight-point paired accuracy gain at 80% power. Rerunning the arithmetic: a five-point gain ($\delta = 0.05$) needs $\approx 783$ items, and a ten-point gain ($\delta = 0.10$) needs only $\approx 194$. These three numbers (194 / 304 / 783 for 10 / 8 / 5 points) are exactly what Chapter 3.9 uses to choose the suite size, trading detectable effect against hand-verification cost.
```

## Tooling

The module could lean entirely on `scipy.stats`, and it uses `scipy` for the t and chi-square distributions. But I wrap it in my own `evalstats` for three reasons that matter to reproducibility. First, the resampling unit in this book is the *item cluster*, and a generic bootstrap does not know that; `evalstats` bootstraps clusters by default. Second, the comparisons are *paired*, and the module makes paired the default so no one accidentally runs an unpaired test on shared items. Third, every result comes back as a typed dataclass with the estimator, the CI, the method, and the seed recorded, so it drops straight into an MLflow run and the number is never bare. A p-value with no method and no seed attached is a rumor; `evalstats` makes the method and seed travel with the number.

## Lab

We build `evalstats`, the module every later comparison in the book imports. It has four estimators (`bootstrap`, `paired`, `correction`, `power`), a result type, and tests that check the bootstrap's coverage by simulation and check McNemar against `scipy`.

### Project setup

```bash title="setup.sh"
uv init evalstats && cd evalstats
uv add "numpy>=1.26" "scipy>=1.13"
uv add --dev "pytest>=8.2"
```

### The result type

```python title="evalstats/result.py"
from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass(frozen=True)
class Estimate:
    """A point estimate with an interval and full provenance. Never bare."""
    name: str
    point: float
    ci_low: float
    ci_high: float
    level: float               # e.g. 0.95
    method: str                # e.g. "cluster_bootstrap_percentile"
    n: int
    n_boot: Optional[int] = None
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)

    def as_mlflow(self) -> dict:
        """Flatten to metric/param dicts for logging (see Ch 3.10)."""
        return {f"{self.name}.point": self.point,
                f"{self.name}.ci_low": self.ci_low,
                f"{self.name}.ci_high": self.ci_high}

    def to_dict(self) -> dict:
        return asdict(self)
```

### The bootstrap

```python title="evalstats/bootstrap.py"
"""Clustered nonparametric bootstrap (Eqs. 7.1-7.4) and paired-difference CI."""
import numpy as np
from typing import Callable, Optional, Sequence
from .result import Estimate

def _resample_indices(rng, groups):
    """Return row indices for one bootstrap resample, clustering by group.
    If groups is None, resample rows i.i.d. (each row is its own cluster)."""
    if groups is None:
        n = groups_n
        return rng.integers(0, n, size=n)  # replaced below
    raise NotImplementedError

def bootstrap_mean(scores: Sequence[float],
                   groups: Optional[Sequence] = None,
                   level: float = 0.95,
                   n_boot: int = 10000,
                   seed: int = 0,
                   name: str = "accuracy") -> Estimate:
    """Percentile bootstrap CI for a mean, optionally clustered by `groups`
    (e.g. item id when there are multiple completions per item)."""
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        n = len(x)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)          # Eq. 7.1
            boot[b] = x[idx].mean()                    # Eq. 7.2
        method = "bootstrap_percentile"
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        buckets = {u: np.where(g == u)[0] for u in uniq}
        k = len(uniq)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            chosen = uniq[rng.integers(0, k, size=k)]  # resample clusters
            idx = np.concatenate([buckets[u] for u in chosen])
            boot[b] = x[idx].mean()
        method = "cluster_bootstrap_percentile"
    lo = float(np.quantile(boot, (1 - level) / 2))     # Eq. 7.4
    hi = float(np.quantile(boot, 1 - (1 - level) / 2))
    return Estimate(name=name, point=float(x.mean()), ci_low=lo, ci_high=hi,
                    level=level, method=method, n=len(x),
                    n_boot=n_boot, seed=seed,
                    extra={"boot_se": float(boot.std(ddof=1))})  # Eq. 7.3

def bootstrap_paired_diff(a: Sequence[float], b: Sequence[float],
                          groups: Optional[Sequence] = None,
                          level: float = 0.95, n_boot: int = 10000,
                          seed: int = 0, name: str = "delta") -> Estimate:
    """Paired-difference CI on shared items: resample items, recompute mean(a-b).
    Assumption-light alternative to the paired t-test (Eq. 7.5)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "paired inputs must share items and order"
    d = a - b
    est = bootstrap_mean(d, groups=groups, level=level, n_boot=n_boot,
                         seed=seed, name=name)
    return est

def bootstrap_statistic(data, stat_fn: Callable, level: float = 0.95,
                        n_boot: int = 10000, seed: int = 0,
                        name: str = "stat") -> Estimate:
    """Bootstrap CI for an arbitrary statistic (e.g. Cohen's kappa from Ch 3.6).
    `data` is a sequence of records; `stat_fn` maps a resampled list -> float."""
    records = list(data)
    n = len(records)
    rng = np.random.default_rng(seed)
    point = stat_fn(records)
    boot = np.empty(n_boot)
    for bi in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[bi] = stat_fn([records[j] for j in idx])
    lo = float(np.quantile(boot, (1 - level) / 2))
    hi = float(np.quantile(boot, 1 - (1 - level) / 2))
    return Estimate(name=name, point=float(point), ci_low=lo, ci_high=hi,
                    level=level, method="bootstrap_percentile", n=n,
                    n_boot=n_boot, seed=seed)
```

### Paired tests

```python title="evalstats/paired.py"
"""Paired t-test (Eq. 7.5) and McNemar's exact/chi-square test (Eqs. 7.6-7.7)."""
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    effect: float          # estimated difference (a - b)
    method: str
    n: int
    extra: dict

def paired_t(a, b, name="paired_t") -> TestResult:
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    t, p = stats.ttest_rel(a, b)          # Eq. 7.5
    return TestResult(name, float(t), float(p), float(d.mean()),
                      "paired_t", len(d), {"sd_diff": float(d.std(ddof=1))})

def mcnemar(a_correct, b_correct, exact=True, name="mcnemar") -> TestResult:
    """Paired binary test. Inputs are 0/1 correctness on shared items."""
    a = np.asarray(a_correct, int); b = np.asarray(b_correct, int)
    b_only = int(np.sum((a == 1) & (b == 0)))   # A right, B wrong
    c_only = int(np.sum((a == 0) & (b == 1)))   # B right, A wrong
    disc = b_only + c_only
    n = len(a)
    if exact:
        # Two-sided exact binomial on discordant pairs (Eq. 7.6).
        p = float(stats.binomtest(min(b_only, c_only), disc, 0.5).pvalue) \
            if disc > 0 else 1.0
        stat = float(min(b_only, c_only)); method = "mcnemar_exact"
    else:
        stat = (abs(b_only - c_only) - 1) ** 2 / disc if disc else 0.0  # Eq.7.7
        p = float(stats.chi2.sf(stat, df=1)); method = "mcnemar_chi2_cc"
    return TestResult(name, stat, p, (b_only - c_only) / n, method, n,
                      {"b": b_only, "c": c_only, "discordant_rate": disc / n})
```

### Corrections and power

```python title="evalstats/correction.py"
"""Bonferroni (Eq. 7.8) and Benjamini-Hochberg (Eq. 7.9)."""
import numpy as np

def bonferroni(pvalues, alpha=0.05):
    p = np.asarray(pvalues, float)
    m = len(p)
    reject = p <= alpha / m
    return {"reject": reject.tolist(), "adjusted": np.minimum(p * m, 1.0).tolist(),
            "alpha": alpha, "m": m, "method": "bonferroni"}

def benjamini_hochberg(pvalues, q=0.05):
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q           # Eq. 7.9 RHS
    below = ranked <= thresh
    kstar = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject = np.zeros(m, bool)
    if kstar > 0:
        reject[order[:kstar]] = True
    # step-up adjusted p-values
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1))
        adj[i] = prev
    adjusted = np.empty(m); adjusted[order] = adj
    return {"reject": reject.tolist(), "adjusted": adjusted.tolist(),
            "k_star": int(kstar), "q": q, "method": "benjamini_hochberg"}
```

```python title="evalstats/power.py"
"""Power / required sample size for paired comparisons (Eqs. 7.10-7.11)."""
import numpy as np
from scipy import stats

def required_n_paired_mean(delta, sd_diff, alpha=0.05, power=0.80):
    """Eq. 7.10: items needed to detect a mean difference `delta`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) * sd_diff / delta) ** 2))

def required_n_mcnemar(delta, psi, alpha=0.05, power=0.80):
    """Eq. 7.11: item pairs needed for a paired accuracy gain `delta`
    given discordant rate `psi` = p10 + p01."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(psi) + z_b * np.sqrt(max(psi - delta ** 2, 0))) ** 2
    return int(np.ceil(num / delta ** 2))
```

```python title="evalstats/__init__.py"
from .result import Estimate
from .bootstrap import (bootstrap_mean, bootstrap_paired_diff,
                        bootstrap_statistic)
from .paired import paired_t, mcnemar, TestResult
from .correction import bonferroni, benjamini_hochberg
from .power import required_n_paired_mean, required_n_mcnemar

__all__ = ["Estimate", "bootstrap_mean", "bootstrap_paired_diff",
           "bootstrap_statistic", "paired_t", "mcnemar", "TestResult",
           "bonferroni", "benjamini_hochberg",
           "required_n_paired_mean", "required_n_mcnemar"]
```

### Tests that keep the module honest

```python title="tests/test_evalstats.py"
import numpy as np
import evalstats as es

def test_bootstrap_coverage():
    """A 95% CI should cover the true mean ~95% of the time (Eq. 7.4)."""
    rng = np.random.default_rng(0)
    p_true = 0.7
    covered = 0
    trials = 300
    for t in range(trials):
        x = rng.binomial(1, p_true, size=200)
        e = es.bootstrap_mean(x, n_boot=2000, seed=t)
        if e.ci_low <= p_true <= e.ci_high:
            covered += 1
    assert 0.90 <= covered / trials <= 0.99  # loose band for a fast test

def test_mcnemar_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.75, 400)
    b = rng.binomial(1, 0.68, 400)
    r = es.mcnemar(a, b, exact=True)
    # discordant-only exact test agrees with a hand binomial
    disc = r.extra["b"] + r.extra["c"]
    p = stats.binomtest(min(r.extra["b"], r.extra["c"]), disc, 0.5).pvalue
    assert abs(r.pvalue - p) < 1e-9

def test_bh_monotone_and_bonferroni_bound():
    p = [0.001, 0.01, 0.03, 0.2, 0.5]
    bh = es.benjamini_hochberg(p, q=0.05)
    bf = es.bonferroni(p, alpha=0.05)
    assert sum(bh["reject"]) >= sum(bf["reject"])  # BH at least as powerful

def test_power_numbers_match_chapter():
    # The three numbers Chapter 3.9 consumes (psi = 0.25).
    assert es.required_n_mcnemar(0.10, 0.25) == 194
    assert es.required_n_mcnemar(0.08, 0.25) == 304
    assert es.required_n_mcnemar(0.05, 0.25) == 783
```

```admonish gotcha
The single most common way to lie to yourself with these tools is to run an *unpaired* test on *paired* data. If model A and model B were evaluated on the same items, comparing their two accuracy CIs and checking whether they overlap is the wrong test and it is badly underpowered, because it ignores the item-difficulty correlation that pairing cancels (the covariance term in the Eq. 7.5 derivation). Two overlapping marginal CIs can still correspond to a strongly significant paired difference. Always feed shared-item comparisons to `bootstrap_paired_diff` or `mcnemar`, never to two separate `bootstrap_mean` calls compared by eye.
```

### The artifact and what you should see

The artifact is the installed `evalstats` module plus a green `pytest`. Run `uv run pytest -q`. You should see four tests pass: the bootstrap's 95% interval covers the true accuracy on the order of 95% of the time (the coverage test), McNemar's exact p-value matches a hand binomial to numerical precision, Benjamini-Hochberg rejects at least as many hypotheses as Bonferroni on the same p-values (BH is uniformly more powerful), and the power module reproduces the exact item counts (194, 304, 783 items for a ten-, eight-, and five-point paired gain at 80% power with a discordant rate of 0.25) that the derivation computed by hand and that Chapter 3.9 will consume to size the thesis suite. From here on, no comparison in this book reports a bare number: an accuracy comes with a `bootstrap_mean` interval, a model-versus-model claim comes with a `bootstrap_paired_diff` or `mcnemar` p-value, a sweep comes with a `benjamini_hochberg` correction, and every one of them carries its method and seed into MLflow. That is the whole point of building the module once, here, and importing it everywhere after.

```admonish read-along
[CAI] is the reference this part leans on for causal thinking, and it pairs naturally with this chapter: a confidence interval tells you whether a difference is real, but Part IV's causal machinery tells you whether that real difference is *attributable to the intervention you made* rather than to a confounder in the eval design. Read this chapter for the "is it noise" question and [CAI] for the "is it causal" question; the thesis needs both answered.
```

```admonish substack-seed
Every benchmark headline you have ever read ("model X beats model Y by 1.3 points") is a claim about the difference of two noisy estimates, and most of them never show the error bars. Here is the uncomfortable arithmetic: at 70% accuracy on 200 items, the confidence interval on a single score is already about plus or minus three points, so a 1.3-point gap is inside the noise before you even ask whether the two runs were paired. The fix is a hundred lines of code you write once: a clustered bootstrap for confidence intervals, McNemar's test for same-items comparisons, Benjamini-Hochberg when you run many tests, and a power calculation that tells you how many items you needed in the first place. Build that module, import it everywhere, and never report a bare eval number again.
```
