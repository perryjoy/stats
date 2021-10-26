import os

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def str_list_in_float(list):
    float_list = []
    for l in list:
        float_list.append(float(l))

    return float_list


def read_file(filename):
    with open(filename) as file:
        values = []
        stop_root = int(file.readline().split(' = ')[1].rstrip())
        for line in file.readlines():
            line = line.rstrip().split('    ')
            line = str_list_in_float(line)
            cell = line[0]
            channels = line[1:]
            values = values + channels

    return values


dirs = ['+0_5V', '+0_25V', '-0_5V', '-0_25V', 'Sin_100MHz', 'ZeroLine']
for dir in dirs:
    values = []
    for addres, dirs, files in os.walk(dir):
        for filename in files:
            values = values + read_file(dir + '/' + filename)
    print(dir + ' in work')
    plt.hist(values, density=True, bins=40)
    plt.savefig(dir+'.png')
    plt.clf()

