import math
import copy
from tqdm import tqdm
import scipy.special as spc

def frequency_block_test(array: list, m: int):
    n = len(array)
    N = math.floor(n / m)
    buf_array = []
    x2 = 0
    new_array = copy.deepcopy(array)
    for i in tqdm(range(0, m * N, m)):
        for j in range(i, i + m):
            buf_array.append(new_array[j])
        a0 = sum(buf_array)
        x2 += ((a0 / m) - (1 / 2)) ** 2
        buf_array.clear()

    result = 1 - spc.gammainc((N / 2), ((x2 * 4 * m) / 2))
    if result >= 0.01:
        return f'------------ \nFrequency Block Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nFrequency Block Test \nUnsuccess P-value = {str(result)} \n------------'
