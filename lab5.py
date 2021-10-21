import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse
from tabulate import tabulate
import statistics

size = [20, 60, 100]
rho = [0, 0.5, 0.9]


def _quadrant(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    x_new = np.empty(size, dtype=float)
    x_new.fill(med_x)
    x_new = x - x_new
    y_new = np.empty(size, dtype=float)
    y_new.fill(med_y)
    y_new = y - y_new
    n = [0, 0, 0, 0]
    for i in range(size):
        if x_new[i] >= 0 and y_new[i] >= 0:
            n[0] += 1
        if x_new[i] < 0 and y_new[i] > 0:
            n[1] += 1
        if x_new[i] < 0 and y_new[i] < 0:
            n[2] += 1
        if x_new[i] > 0 and y_new[i] < 0:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / size


def _find_coefficients_s_p(size_, rho):
    rv_mean = [0, 0]
    rv_cov = [[1.0, rho], [rho, 1.0]]
    pearson_coef = np.empty(1000, dtype=float)
    spearman_coef = np.empty(1000, dtype=float)
    quadrant_coef = np.empty(1000, dtype=float)
    for i in range(1000):
        rv = stats.multivariate_normal.rvs(rv_mean, rv_cov, size=size_)
        x = rv[:, 0]
        y = rv[:, 1]
        pearson_coef[i], t = stats.pearsonr(x, y)
        spearman_coef[i], t = stats.spearmanr(x, y)
        quadrant_coef[i] = _quadrant(x, y)
    return pearson_coef, spearman_coef, quadrant_coef


def _create_table(pearson_coef, spearman_coef, quadrant_coef, rho, size):
    rows = []
    headers = []
    if rho != -1:
        rows.append(["rho = " + str(rho), 'r', 'r_{S}', 'r_{Q}'])
    else:
        rows.append(["n = " + str(size), 'r', 'r_{S}', 'r_{Q}'])
    p = np.median(pearson_coef)
    s = np.median(spearman_coef)
    q = np.median(quadrant_coef)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = np.median([pearson_coef[k] ** 2 for k in range(1000)])
    s = np.median([spearman_coef[k] ** 2 for k in range(1000)])
    q = np.median([quadrant_coef[k] ** 2 for k in range(1000)])
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = statistics.variance(pearson_coef)
    s = statistics.variance(spearman_coef)
    q = statistics.variance(quadrant_coef)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    print(tabulate(rows, headers, tablefmt="latex"))
    print('\n')


def _confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _scatter(size):
    mean = [0, 0]
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("n = " + str(size))
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    num = 0
    for r in rho:
        cov = [[1.0, r], [r, 1.0]]
        rv = stats.multivariate_normal.rvs(mean, cov, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[num].scatter(x, y, s=3)
        _confidence_ellipse(x, y, ax[num], edgecolor='navy')
        ax[num].scatter(np.mean(x), np.mean(y), c='aqua', s=3)
        ax[num].set_title(titles[num])
        num += 1
    plt.savefig("n" + str(size) + ".png", format='png')
    plt.show()


if __name__ == '__main__':
    for j in size:
        for i in rho:
            pearson_coef, spearman_coef, quadrant_coef =  _find_coefficients_s_p(j, i)
            _create_table(pearson_coef, spearman_coef, quadrant_coef, i, j)
        pearson_coef = np.empty(1000, dtype=float)
        spearman_coef = np.empty(1000, dtype=float)
        quadrant_coef = np.empty(1000, dtype=float)
        for k in range(1000):
            rv = []
            for l in range(2):
                x = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], j) + 0.1 * stats. multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], j)
                rv += list(x)
            rv = np.array(rv)
            x = rv[:, 0]
            y = rv[:, 1]
            pearson_coef[k], t = stats.pearsonr(x, y)
            spearman_coef[k], t = stats.spearmanr(x, y)
            quadrant_coef[k] = _quadrant(x, y)
        _create_table(pearson_coef, spearman_coef, quadrant_coef, -1, j)
        _scatter(j)












