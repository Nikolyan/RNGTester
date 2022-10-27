from math import floor
from tqdm import tqdm
import scipy.special as spc

def frequency_block_test(bin_data: list, path: str, m: int):
    n = len(bin_data)
    N = floor(n / m)
    buf_array = []
    x2 = 0
    for i in tqdm(range(0, m * N, m)):
        for j in range(i, i + m):
            buf_array.append(bin_data[j])
        a0 = sum(buf_array)
        x2 += ((a0 / m) - (1 / 2)) ** 2
        buf_array.clear()

    result = 1 - spc.gammainc((N / 2), ((x2 * 4 * m) / 2))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nFrequency Block Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nFrequency Block Test \nUnsuccess P-value = {str(result)} \n------------\n')

    return 0
