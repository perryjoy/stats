import numpy as np
from tabulate import tabulate


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




def quart(array, p):
    new_array = np.sort(array)
    k = len(array) * p
    if k.is_integer():
        return new_array[int(k)]
    else:
        return new_array[int(k) + 1]


def z_q(array):
    low_q = 0.25
    high_q = 0.75
    return (quart(array, low_q) + quart(array, high_q)) / 2


def z_tr(array):
    r = int(len(array) * 0.25)
    new_array = np.sort(array)
    sum = 0
    for i in range(r + 1, len(array) - r):
        sum += new_array[i]
    return sum / (len(array) - 2 * r)




quan_of_numbers = [10, 100, 1000]

repeat = 1000
if __name__ == '__main__':
    for dist_name in distr_type:
        headers = [dist_name, "z_", "med", "z_r", "z_q", "z_tr"]
        i = 0
        rows = []
        for N in quan_of_numbers:
            mean = []
            med = []
            z_r = []
            z_q_ = []
            z_tr_ = []
            for rep in range(0, repeat):
                array = get_distr_samples(dist_name, N)
                array_sorted = np.sort(array)
                mean.append(np.mean(array))
                med.append(np.median(array))
                z_r.append((array_sorted[0] + array_sorted[-1]) / 2)
                z_q_.append(z_q(array))
                z_tr_.append(z_tr(array))
            rows.append([" E(z) " + str(N),
                              np.around(np.mean(mean), decimals=6),
                              np.around(np.mean(med), decimals=6),
                              np.around(np.mean(z_r), decimals=6),
                              np.around(np.mean(z_q_), decimals=6),
                              np.around(np.mean(z_tr_), decimals=6)])
            rows.append([" D(z) " + str(N),
                         np.around(np.std(mean) * np.std(mean), decimals=6),
                         np.around(np.std(med) * np.std(med), decimals=6),
                         np.around(np.std(z_r) * np.std(z_r), decimals=6),
                         np.around(np.std(z_q_) * np.std(z_q_), decimals=6),
                         np.around(np.std(z_tr_) * np.std(z_tr_), decimals=6)])
            i += 1
        print(tabulate(rows, headers, tablefmt="latex"))
        print("\n")
