from scipy.stats import norm
import numpy as np

import matplotlib.pyplot as plt


def bland_altman_plot(x, y, title,ax):
    mean = np.mean([x, y], axis=0)
    diff = x - y
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    ax.scatter(mean, diff)
    ax.axhline(md, color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Mean of two measurements')
    ax.set_ylabel('Difference')
    return ax
    
    
def mean_ci(series, alpha=0.05):
    n = series.count()
    mean = series.mean()
    std = series.std()
    se = std / np.sqrt(n)
    ci = norm.interval(1-alpha, loc=mean, scale=se)
    return mean, std, n, ci

def cohen_d(x, y):
  
    nx, ny = x.count(), y.count()
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled_sd = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    d = (my - mx) / pooled_sd
 
    se_d = np.sqrt((nx + ny) / (nx * ny) + d**2 / (2*(nx + ny)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    return d, (ci_low, ci_high)


