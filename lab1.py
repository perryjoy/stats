import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


quan_of_numbers = [10, 50, 1000]
mu, sigma = 0, 1
num_bins = 20
i = 1

plt.suptitle('Normal Distribution')
for N in quan_of_numbers:
    plt.subplot(1, 3, i)
    s = np.random.normal(mu, sigma, N)
    n, bins, patches = plt.hist(s, num_bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - mu)**2 / (2 * sigma**2) ), color='b', linewidth=1)
    plt.title(r'Normal Distribution: $\mu=0$, $\sigma=1$, N=%i' %N)
    plt.xlabel('NormalNumbers')
    plt.ylabel('Density')
    i += 1
    plt.subplots_adjust(wspace=0.5)
plt.show()

i = 1
plt.suptitle('Cauchy Distribution')
for N in quan_of_numbers:
    plt.subplot(1, 3, i)
    s = np.random.standard_cauchy(N)
    n, bins, patches = plt.hist(s, num_bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
    plt.plot(bins, 1 / (np.pi * (bins * bins + 1)), color='b', linewidth=1)
    plt.title(r'Cauchy Distribution: N=%i' %N)
    plt.xlabel('Numbers')
    plt.ylabel('Density')
    i += 1
    plt.subplots_adjust(wspace=0.5)
plt.show()

i = 1
mu, sigma = 0, np.sqrt(2)
plt.suptitle('Laplace Distribution')
for N in quan_of_numbers:
    plt.subplot(1, 3, i)
    s = np.random.laplace(mu, sigma, N)
    n, bins, patches = plt.hist(s, num_bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
    plt.plot(bins, 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(bins)), color='b', linewidth=1)
    plt.title(r'Laplace Distribution: N=%i' %N)
    plt.xlabel('LaplaceNumbers')
    plt.ylabel('Density')
    i += 1
    plt.subplots_adjust(wspace=0.5)
plt.show()

i = 1
plt.suptitle('Poisson Distribution')
for N in quan_of_numbers:
    plt.subplot(1, 3, i)
    s = np.random.poisson(10, N)
    n, bins, patches = plt.hist(s, num_bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
    plt.plot(bins, np.power(10, bins) * np.exp(-10) / factorial(bins), color='b', linewidth=1)
    plt.title(r'Poisson Distribution: N=%i' %N)
    plt.xlabel('PoissonNumbers')
    plt.ylabel('Density')
    i += 1
    plt.subplots_adjust(wspace=0.5)
plt.show()

i = 1
plt.suptitle('Uniform Distribution')
for N in quan_of_numbers:
    plt.subplot(1, 3, i)
    s = np.random.uniform(-np.sqrt(3), np.sqrt(3), N)
    n, bins, patches = plt.hist(s, num_bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
    pdf = []
    ar = np.arange(-2., 2., 0.01)
    for elem in ar:
        pdf.append(1 / (2 * np.sqrt(3))) if np.fabs(elem) <= np.sqrt(3) else pdf.append(0)
    plt.plot(ar, pdf, color='b', linewidth=1)
    plt.title(r'Uniform Distribution, N=%i' %N)
    plt.xlabel('UniformNumbers')
    plt.ylabel('Density')
    i += 1
    plt.subplots_adjust(wspace=0.5)
plt.show()
