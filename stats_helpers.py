from scipy.stats import norm
import numpy as np



def mean_ci(series, alpha=0.05):
    n = series.count()
    mean = series.mean()
    std = series.std()
    se = std / np.sqrt(n)
    ci = norm.interval(1-alpha, loc=mean, scale=se)
    return mean, std, n, ci

def cohen_d(x, y):
    # MATLAB style: pooled std with denominator n-1 for each group (unbiased)
    nx, ny = x.count(), y.count()
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled_sd = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    d = abs(mx - my) / pooled_sd
    # 95% CI for Cohen's d (Hedges & Olkin, 1985, as in MATLAB)
    se_d = np.sqrt((nx + ny) / (nx * ny) + d**2 / (2*(nx + ny)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    return d, (ci_low, ci_high)

