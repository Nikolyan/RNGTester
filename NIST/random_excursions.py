from tqdm import trange
import scipy.special as spc
from numpy import zeros, cumsum, abs, append, array, where, transpose, sum, clip


def value_of_pik(k, x):
    if k == 0:
        out = 1 - 1.0 / (2 * abs(x))
    elif k >= 5:
        out = (1.0 / (2 * abs(x))) * (1 - 1.0 / (2 * abs(x))) ** 4
    else:
        out = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * abs(x))) ** (k - 1)
    return out


def random_excursions(bin_data: list, path: str):
    int_data = zeros(len(bin_data))
    for i in trange(len(bin_data)):
        if bin_data[i] == 0:
            int_data[i] = -1.0
        else:
            int_data[i] = 1.0
    cumul_sum = cumsum(int_data)
    cumul_sum = append(cumul_sum, [0])
    cumul_sum = append([0], cumul_sum)
    values_of_x = array([-4, -3, -2, -1, 1, 2, 3, 4])
    position = where(cumul_sum == 0)[0]
    array_of_cycles = []
    for pos in range(len(position) - 1):
        array_of_cycles.append(cumul_sum[position[pos]:position[pos + 1] + 1])
    num_cycles = len(array_of_cycles)
    counter = []
    for cycle in array_of_cycles:
        counter.append(([len(where(cycle == state)[0]) for state in values_of_x]))
    counter = transpose(clip(counter, 0, 5))
    s_1 = []
    for cycle in range(6):
        s_1.append([(sct == cycle).sum() for sct in counter])
    s_1 = transpose(s_1)
    piks = ([([value_of_pik(uu, state) for uu in range(6)]) for state in values_of_x])
    inner_term = num_cycles * array(piks)
    chi = sum(1.0 * (array(s_1) - inner_term) ** 2 / inner_term, axis=1)
    result = ([spc.gammaincc(2.5, cs / 2.0) for cs in chi])
    for i in range(len(values_of_x)):
        if result[i] >= 0.01:
            open(path, 'a').write(
                f'------------\nRandom Excursions Test\nSuccess P-value = {str(result[i])}, State = {values_of_x[i]}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------\nRandom Excursions Test\nUnsuccess P-value = {str(result[i])}, State = {values_of_x[i]}\n------------\n')
    return 0