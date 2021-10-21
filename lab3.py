import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.special as sc



distr_type = ['Norm', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


def get_distr_samples(d_name, num):
    if d_name == 'Norm':
        return np.random.normal(0, 1, num)
    elif d_name == 'Cauchy':
        return np.random.standard_cauchy(num)
    elif d_name == 'Laplace':
        return np.random.laplace(0, np.sqrt(2) / 2, num)
    elif d_name == 'Poisson':
        return np.random.poisson(10, num)
    elif d_name == 'Uniform':
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), num)
    return []


def get_distr_func(d_name, x):
    if d_name == 'Norm':
        return 0.5 * (1 + sc.erf(x / np.sqrt(2)))
    elif d_name == 'Cauchy':
        return np.arctan(x) / np.pi + 0.5
    elif d_name == 'Laplace':
        if x <= 0:
            return 0.5 * np.exp(np.sqrt(2) * x)
        else:
            return 1 - 0.5 * np.exp(-np.sqrt(2) * x)
    elif d_name == 'Poisson':
        return sc.gammainc(x + 1, 10) / sc.factorial(x)
    elif d_name == 'Uniform':
        if x < -np.sqrt(3):
            return 0
        elif np.fabs(x) <= np.sqrt(3):
            return (x - np.sqrt(3)) / 2 * np.sqrt(3)
        else:
            return 1
    return 0


def quart(array, p):
    new_array = np.sort(array)
    k = len(array) * p
    if k.is_integer():
        return new_array[int(k)]
    else:
        return new_array[int(k) + 1]



quan_of_numbers = [20, 100]
repeat = 1000

if __name__ == '__main__':
    headers = ["distribution name", "proportion of ejections"]
    headers_th = ["distibution name", "q_1", "q_3", "x_1", "x_2", "p"]
    rows = []
    rows_th = []
    for dist_name in distr_type:
        array_20 = get_distr_samples(dist_name, quan_of_numbers[0])
        array_100 = get_distr_samples(dist_name, quan_of_numbers[1])
        line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
        bbox_props = dict(color="b", alpha=0.9)
        flier_props = dict(marker="o", markersize=4)
        plt.boxplot((array_20, array_100), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
        plt.ylabel("X")
        plt.title(dist_name)
        #plt.savefig(dist_name + '.png', format='png')
        #plt.show()



    #for dist_name in distr_type:
        count = [0, 0]

        for i in range (0, repeat):
            array_20 = get_distr_samples(dist_name, quan_of_numbers[0])
            array_100 = get_distr_samples(dist_name, quan_of_numbers[1])

            X_20 = []
            X_100 = []

            X_20.append(np.quantile(array_20, 0.25) - 1.5 * (np.quantile(array_20, 0.75) - np.quantile(array_20, 0.25)))
            X_20.append(np.quantile(array_20, 0.75) + 1.5 * (np.quantile(array_20, 0.75) - np.quantile(array_20, 0.25)))

            X_100.append(np.quantile(array_100, 0.25) - 1.5 * (np.quantile(array_100, 0.75) - np.quantile(array_100, 0.25)))
            X_100.append(np.quantile(array_100, 0.75) + 1.5 * (np.quantile(array_100, 0.75) - np.quantile(array_100, 0.25)))


            for k in range(0, 20):
                if array_20[k] > X_20[1] or array_20[k] < X_20[0]:
                    count[0] = count[0] + 1

            for k in range(0, 100):
                if array_100[k] > X_100[1] or array_100[k] < X_100[0]:
                    count[1] = count[1] + 1

        count[1] /= 1000
        count[0] /= 1000
        rows.append([dist_name + ", n = 20",  np.around(count[0] / 20, decimals=3)])
        rows.append([dist_name + ", n = 100", np.around(count[1] / 100, decimals=3)])

    print(tabulate(rows, headers, tablefmt="latex"))
    print("\n")


    print(tabulate(rows_th, headers_th, tablefmt="latex"))
    print("\n")



