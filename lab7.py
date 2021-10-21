import numpy as np
from tabulate import tabulate
import scipy.stats as stats

if __name__ == '__main__':
    distr = np.random.normal(0, 1, size=100)
    mu_n = np.mean(distr)
    sigma_n = np.std(distr)
    print(np.around(mu_n, decimals=2), ' ', np.around(sigma_n, decimals=2))

    alpha = 0.05
    p = 1 - alpha
    k = 6

    limits = np.linspace(-1.2, 1.2, num=k-1)
    sample = stats.chi2.ppf(p, k-1)
    array = np.array([stats.norm.cdf(limits[0])])
    quan_ar = np.array([len(distr[distr <= limits[0]])])
    for i in range(0, len(limits) - 1):
        new_ar = stats.norm.cdf(limits[i + 1]) - stats.norm.cdf(limits[i])
        array = np.append(array, new_ar)
        quan_ar = np.append(quan_ar, len(distr[(distr <= limits[i + 1]) & (distr >= limits[i])]))
    array = np.append(array, 1 - stats.norm.cdf(limits[-1]))
    quan_ar = np.append(quan_ar, len(distr[distr >= limits[-1]]))
    result = np.divide(np.multiply((quan_ar - 100 * array), (quan_ar - 100 * array)), array * 100)

    headers = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "...^2"]
    rows = []
    for i in range(0, len(quan_ar)):
        if i == 0:
            boarders = ['-inf', np.around(limits[0], decimals=2)]
        elif i == len(quan_ar) - 1:
            boarders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]
        rows.append([i + 1, boarders, quan_ar[i], np.around(array[i], decimals=4), np.around(array[i] * 100, decimals = 2),
                     np.around(quan_ar[i] - 100 * array[i], decimals=2), np.around(result[i], decimals=2)])
    rows.append([len(quan_ar), "-", np.sum(quan_ar), np.around(np.sum(array), decimals=4),
                 np.around(np.sum(array * 100), decimals=2),
                 -np.around(np.sum(quan_ar - 100 * array), decimals=2),
                 np.around(np.sum(result), decimals=2)])
    print(tabulate(rows, headers, tablefmt="latex"))

    print(len(quan_ar))
    print('\n')




