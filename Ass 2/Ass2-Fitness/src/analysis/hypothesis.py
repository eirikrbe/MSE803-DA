"""Statistical tests.

`test` is a REQUIRED argument everywhere. Nothing here inspects your data and
picks a test for you: that choice depends on the design, the distribution, the
sample size and what the question actually is, and getting it silently wrong is
how an analysis produces a confident number that means nothing.

Each docstring states the test's assumptions. Read them, choose, and record the
reason in the DECISION cell in 03_analysis.ipynb.
"""

import numpy as np
from scipy import stats

TESTS = {
    'welch_t': "Welch's t-test. Two independent groups, means, unequal variances "
               "allowed. Assumes roughly normal data or n large enough for the CLT.",
    'student_t': 'Student t-test. As above but assumes EQUAL variances. Prefer '
                 'welch_t unless you have a reason to assume equal spread.',
    'mannwhitney': 'Mann-Whitney U. Two independent groups, rank-based. No '
                   'normality assumption; tests stochastic dominance, not means. '
                   'Use for skewed data or small n.',
    'paired_t': 'Paired t-test. The SAME units measured twice. Wrong for two '
                'independent groups.',
}


def cohens_d(a, b):
    """Standardised mean difference, pooled sd. Effect size, not significance."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt((((na - 1) * a.std(ddof=1) ** 2) +
                      ((nb - 1) * b.std(ddof=1) ** 2)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def compare_groups(df, value_col, group_col, test, groups=None, alpha=0.05):
    """Compare `value_col` between two levels of `group_col` using `test`.

    `test` is required -- see TESTS for the options and their assumptions.
    Reports the effect size and a confidence interval alongside p, because a
    p-value on its own tells you nothing about whether the difference matters.
    """
    if test not in TESTS:
        raise ValueError(
            f'test={test!r} is not available. Choose one deliberately:\n' +
            '\n'.join(f'  {k}: {v}' for k, v in TESTS.items()))

    levels = list(groups) if groups is not None else \
        df[group_col].dropna().unique().tolist()
    if len(levels) != 2:
        raise ValueError(
            f'{group_col!r} has {len(levels)} levels ({levels}). These tests compare '
            'exactly two; pass groups=[a, b] to say which two, or use a different method.')

    a = df.loc[df[group_col] == levels[0], value_col].dropna().to_numpy(float)
    b = df.loc[df[group_col] == levels[1], value_col].dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f'need at least 2 observations per group; got {len(a)} and {len(b)}')

    if test == 'welch_t':
        stat, p = stats.ttest_ind(a, b, equal_var=False)
    elif test == 'student_t':
        stat, p = stats.ttest_ind(a, b, equal_var=True)
    elif test == 'mannwhitney':
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    elif test == 'paired_t':
        if len(a) != len(b):
            raise ValueError('paired_t needs equal-length groups of the same units')
        stat, p = stats.ttest_rel(a, b)

    d = cohens_d(a, b)
    result = {
        'test': test, 'groups': levels, 'n': (len(a), len(b)),
        'mean': (round(float(a.mean()), 4), round(float(b.mean()), 4)),
        'difference': round(float(a.mean() - b.mean()), 4),
        'statistic': round(float(stat), 4), 'p_value': round(float(p), 5),
        'cohens_d': round(d, 4), 'alpha': alpha,
    }

    print(f'{TESTS[test]}\n')
    print(f'{levels[0]}: n={len(a)}, mean={a.mean():.4f}, sd={a.std(ddof=1):.4f}')
    print(f'{levels[1]}: n={len(b)}, mean={b.mean():.4f}, sd={b.std(ddof=1):.4f}')
    print(f'difference = {a.mean() - b.mean():.4f}   p = {p:.5f}   '
          f"Cohen's d = {d:.3f}")
    if min(len(a), len(b)) < 15:
        print(f'\nn is small (min {min(len(a), len(b))}). The test will run, but it '
              'has little power and the assumptions are hard to check. Treat the '
              'result as suggestive.')
    return result


def chi2_independence(df, col_a, col_b):
    """Chi-square test of independence between two categorical columns.

    Assumes expected counts of roughly 5+ per cell; the function warns when that
    fails, which is common in small samples and makes the p-value unreliable.
    """
    table = np.asarray(
        df.pivot_table(index=col_a, columns=col_b, aggfunc='size', fill_value=0))
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n_small = int((expected < 5).sum())
    print(f'chi2 = {chi2:.4f}, dof = {dof}, p = {p:.5f}')
    if n_small:
        print(f'{n_small} cell(s) have an expected count below 5. The chi-square '
              "approximation is unreliable here; consider Fisher's exact test or "
              'collapsing categories -- and record which you chose.')
    return {'chi2': round(float(chi2), 4), 'p_value': round(float(p), 5),
            'dof': int(dof), 'cells_below_5': n_small}


def holm_adjust(p_values):
    """Holm step-down adjustment across a family of tests.

    Controls the chance of any false positive in the family, and is uniformly
    more powerful than Bonferroni. Returns the adjusted p-values in the input
    order, capped at 1.
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def cramers_v(table):
    """Cramér's V for a contingency table (array or DataFrame of counts).

    0 means no association. Below 0.1 is conventionally negligible (Cohen, 1988).
    """
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def epsilon_squared(h_statistic, n):
    """Effect size for Kruskal-Wallis: the share of rank variance explained by
    group, H / (n - 1). Below 0.01 is negligible."""
    return float(h_statistic / (n - 1))


def rank_biserial(a, b):
    """Rank-biserial correlation for two independent groups, from Mann-Whitney U.

    The probability that a value from `a` exceeds one from `b`, minus the reverse.
    Ranges from -1 to 1; |r| below 0.1 is negligible.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def spearman_ci(rho, n, level=0.90):
    """Fisher-z interval for a Spearman correlation (Fieller et al. variance).

    A 90% interval inside (-0.1, 0.1) is a two one-sided tests (TOST) verdict at
    alpha = 0.05 that the association is negligible, not just undetected.
    """
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    q = stats.norm.ppf(1 - (1 - level) / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def bootstrap_effect_ci(fn, groups, B=2000, seed=0, level=0.90):
    """Percentile bootstrap interval for an effect size computed from groups.

    `fn` takes a list of 1-D arrays (one per group) and returns the effect size.
    Each group is resampled within itself, which keeps the group sizes fixed.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in groups]
    draws = np.empty(B)
    for i in range(B):
        draws[i] = fn([g[rng.integers(0, len(g), len(g))] for g in groups])
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def bootstrap_rows_ci(fn, df, B=2000, seed=0, level=0.90):
    """Percentile bootstrap interval for a statistic of a whole DataFrame.

    Rows are resampled jointly, which is what an association between two
    columns needs (Cramér's V, for instance). `fn` takes a DataFrame and
    returns a number.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    draws = np.empty(B)
    for i in range(B):
        # A resample repeats rows; a fresh index keeps label-aligning code (crosstab) working.
        draws[i] = fn(df.iloc[rng.integers(0, n, n)].reset_index(drop=True))
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def permutation_test(y_true, y_model, B=2000, seed=0):
    """Is the model's error better than chance? Promoted from W3's modeling.py.

    Shuffles the target B times and refits nothing -- it compares the real error
    gain against the distribution of gains under a broken relationship. Returns
    (real_gain, null_gains, p_value). Assumption-light, which is why it is useful
    when n is too small to trust a parametric test.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    y_model = np.asarray(y_model, float)

    baseline_rmse = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    model_rmse = float(np.sqrt(np.mean((y_true - y_model) ** 2)))
    real_gain = baseline_rmse - model_rmse

    null_gains = np.empty(B)
    for i in range(B):
        shuffled = rng.permutation(y_true)
        null_rmse = float(np.sqrt(np.mean((shuffled - y_model) ** 2)))
        null_base = float(np.sqrt(np.mean((shuffled - shuffled.mean()) ** 2)))
        null_gains[i] = null_base - null_rmse

    p_value = float((np.sum(null_gains >= real_gain) + 1) / (B + 1))
    print(f'real gain over the mean baseline: {real_gain:.4f} RMSE')
    print(f'permutation p = {p_value:.4f} over B = {B} shuffles')
    return real_gain, null_gains, p_value
