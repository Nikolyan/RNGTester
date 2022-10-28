from tqdm import trange
import scipy.special as spc
from numpy import zeros

def serial(bin_data: str, path: str, pattern_length=16):
    n = len(bin_data)
    bin_data += bin_data[:pattern_length - 1:]
    array_max = ''
    for i in range(pattern_length + 1):
        array_max += '1'
    vobs1 = zeros(int(array_max[0:pattern_length:], 2) + 1)
    vobs2 = zeros(int(array_max[0:pattern_length - 1:], 2) + 1)
    vobs3 = zeros(int(array_max[0:pattern_length - 2:], 2) + 1)

    for i in trange(n):
        vobs1[int(bin_data[i:i + pattern_length:], 2)] += 1
        vobs2[int(bin_data[i:i + pattern_length - 1:], 2)] += 1
        vobs3[int(bin_data[i:i + pattern_length - 2:], 2)] += 1

    all_vobs = [vobs1, vobs2, vobs3]
    sums = zeros(3)
    for i in range(3):
        for j in range(len(all_vobs[i])):
            sums[i] += ((all_vobs[i][j])**2)
        sums[i] = (sums[i] * (2**(pattern_length - i)) / n) - n
    del1 = sums[0] - sums[1]
    result = spc.gammaincc((2**(pattern_length - 1)) / 2, del1 / 2.0)
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nSerial Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nSerial Test\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0
