import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns


distr_type = ['Norm', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


def get_distr_samples(d_name, num):
    if d_name == 'Norm':
        return stats.norm.rvs(0, 1, size=num)
    elif d_name == 'Cauchy':
        return stats.cauchy.rvs(size=num)
    elif d_name == 'Laplace':
        return stats.laplace.rvs(0, 1 / (2 ** 0.5), size=num)
    elif d_name == 'Poisson':
        return stats.poisson.rvs(10,size=num)
    elif d_name == 'Uniform':
        mu = float(3 ** 0.5)
        return stats.uniform.rvs(-mu, 2 * mu, size=num)
    return []


def get_distr_func(d_name, array):
    if d_name == 'Norm':
        return stats.norm.cdf(array)
    elif d_name == 'Cauchy':
        return stats.cauchy.cdf(array)
    elif d_name == 'Laplace':
        return stats.laplace.cdf(array)
    elif d_name == 'Poisson':
        return stats.poisson.cdf(array, 10)
    elif d_name == 'Uniform':
        return stats.uniform.cdf(array)
    return []


def get_distr_density(d_name, array):
    if d_name == 'Norm':
        return stats.norm.pdf(array, 0, 1)
    elif d_name == 'Cauchy':
        return stats.cauchy.pdf(array)
    elif d_name == 'Laplace':
        return stats.laplace.pdf(array, 0, 1 / 2 ** 0.5)
    elif d_name == 'Poisson':
        return stats.poisson.pmf(10, array)
    elif d_name == 'Uniform':
        mu = float(3 ** 0.5)
        return stats.uniform.pdf(array, -mu, 2 * mu)
    return []

quan_of_numbers = [20, 60, 100]
repeat = 1000

if __name__ == '__main__':
    for dist_name in distr_type:
        if dist_name != "Poisson":
            array_global = np.arange(-4, 4, 0.01)
        else:
            array_global = np.arange(0, 20, 1)
        array_20 = get_distr_samples(dist_name, quan_of_numbers[0])
        array_60 = get_distr_samples(dist_name, quan_of_numbers[1])
        array_100 = get_distr_samples(dist_name, quan_of_numbers[2])
        arrays = [array_20, array_60, array_100]


        j = 1
        for array in arrays:
            frequency = {}
            for i in array:
                if dist_name == "Poisson":
                    if i < 6 or i > 14:
                        array = np.delete(array, list(array).index(i))
                else:
                    if i < -4 or i > 4:
                        array = np.delete(array, list(array).index(i))
                if list(frequency.values()).__contains__(i):
                    frequency[i] = frequency[i] + 1
                    continue
                else:
                    frequency.update([(i, 0)])
            plt.subplot(1, 3, j)
            plt.title(dist_name + ', n = ' + str(quan_of_numbers[j - 1]))
            plt.plot(array_global, get_distr_func(dist_name, array_global), color='blue', linewidth=0.8)
            if dist_name != 'Poisson':
                array_ex = np.linspace(-4, 4)
            else:
                array_ex = np.linspace(6, 14)
            ecdf = ECDF(array)
            y = ecdf(array_ex)
            plt.step(array_ex, y, color='black')
            plt.xlabel('x')
            plt.ylabel('F(x)')
            plt.subplots_adjust(wspace=0.5)
            j = j + 1
        plt.savefig(dist_name + '.png', format='png')
        plt.show()

        k = 1
        for array in arrays:
            for i in array:
                if dist_name == "Poisson":
                    if i < 6 or i > 14:
                        array = np.delete(array, list(array).index(i))
                else:
                    if i < -4 or i > 4:
                        array = np.delete(array, list(array).index(i))
            titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
            l = 0
            fig, ax = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=0.5)
            for bandwidth in [0.5, 1, 2]:
                kde = stats.gaussian_kde(array, bw_method='silverman')
                h_n = kde.factor
                fig.suptitle(dist_name + ', n = ' + str(quan_of_numbers[k - 1]))
                ax[l].plot(array_global, get_distr_density(dist_name, array_global), color='blue', alpha=0.5, label='density')
                ax[l].set_title(titles[l])
                sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
                ax[l].set_xlabel('x')
                ax[l].set_ylabel('f(x)')
                ax[l].set_ylim([0, 1])
                if dist_name == 'Poisson':
                    ax[l].set_xlim([6, 14])
                else:
                    ax[l].set_xlim([-4, 4])
                ax[l].legend()
                l = l + 1
            plt.savefig(dist_name + ' n = ' + str(quan_of_numbers[k-1]) +'.png', format='png')
            plt.show()
            k = k + 1


