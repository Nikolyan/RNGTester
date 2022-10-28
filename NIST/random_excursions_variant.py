from tqdm import trange
import scipy.special as spc
from numpy import zeros, ones, cumsum, abs, where, sqrt

def frequency_fun(list_data, trigger):
    frequency = 0
    for (x, y) in list_data:
        if x == trigger:
            frequency = y
    return frequency


def random_excursions_variant(bin_data: list, path: str):
    int_data_zeros = zeros(len(bin_data))
    for i in trange(len(bin_data)):
        int_data_zeros[i] = bin_data[i]
    cumul_sum = cumsum((2 * int_data_zeros) - ones(len(int_data_zeros)))

    list_data = []
    for j in sorted(set(cumul_sum)):
        if abs(j) <= 9:
            list_data.append([j, len(where(cumul_sum == j)[0])])

    j = frequency_fun(list_data, 0) + 1
    for j in range(-9, 9 + 1):
        if not j == 0:
            den = sqrt(2 * j * (4 * abs(j) - 2))
            result = spc.erfc(abs(frequency_fun(list_data, j) - j) / den)
            if result >= 0.01:
                open(path, 'a').write(
                    f'------------\nRandom Excursions Variant Test {j}\nSuccess P-value = {str(result)}\n------------\n')
            else:
                open(path, 'a').write(
                    f'------------\nRandom Excursions Variant Test {j}\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0
