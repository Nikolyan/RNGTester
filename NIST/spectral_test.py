from math import log
from tqdm import tqdm
import scipy.special as spc
from scipy import fft
from numpy import abs, where, sqrt


def spectral_test(bin_data: list, path: str):
    n = len(bin_data)
    positive_negative_array = []
    for char in tqdm(bin_data):
        if char == 0:
            positive_negative_array.append(-1)
        elif char == 1:
            positive_negative_array.append(1)
    s = fft.fft(positive_negative_array)
    tau = sqrt(log(1 / 0.05) * n)
    count_n0 = 0.95 * (n / 2)
    modulus = abs(s[0: int(n / 2)])
    count_n1 = len(where((modulus[0:len(modulus)] < tau) & (modulus[0:len(modulus)] > 0))[0])
    d = (count_n1 - count_n0) / (sqrt((n * 0.95 * 0.05) / 4))
    result = spc.erfc(abs(d) / sqrt(2))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nSpectral Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nSpectral Test\nUnsuccess P-value = {str(result)}\n------------\n')

    return 0
