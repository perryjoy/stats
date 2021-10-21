import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt


def _mnk_coefficients(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def _fun_for_minim(params, x, y):
    a_1, a_2 = params
    res = 0
    for i in range(len(x)):
        res += abs(a_1 * x[i] + a_2 - y[i])
    return res


def _plot_regr(x, y, type):
    beta_0, beta_1 = _mnk_coefficients(x, y)
    print('MNK')
    print('beta_0 = ' + str(np.around(beta_0, decimals=2)))
    print('beta_1 = ' + str(np.around(beta_1, decimals=2)))
    result = opt.minimize(_fun_for_minim, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    a_0, a_1 = coefs[0], coefs[1]
    print('MNA')
    print('a_0 = ' + str(np.around(a_0, decimals=2)))
    print('a_1 = ' + str(np.around(a_1, decimals=2)))
    plt.scatter(x[1:-2], y[1:-2], label='Выборка', edgecolor='navy')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='blue')
    plt.plot(x, x * (beta_1 * np.ones(len(x))) + beta_0 * np.ones(len(x)), label='МHK', color='deepskyblue')
    plt.plot(x, x * (a_1 * np.ones(len(x))) + a_0 * np.ones(len(x)), label='МHM', color='indigo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.title(type)
    plt.savefig(type + '.png', format='png')
    plt.show()


if __name__ == '__main__':
    x = np.arange(-1.8, 2, 0.2)
    y = 2 * x + 2 * np.ones(len(x)) + np.random.normal(0, 1, size=len(x))
    types = ['Без возмущений', 'С возмущениями']
    _plot_regr(x, y, types[0])
    y[0] += 10
    y[-1] -= 10
    _plot_regr(x, y, types[1])

