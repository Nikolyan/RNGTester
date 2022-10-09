import math
from tqdm import trange, tqdm
import scipy.special as spc
import numpy.fft as sff
import numpy

def spectral_test(bin_data: str):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the peak heights in the Discrete Fourier Transform of the sequence. The purpose of
    this test is to detect periodic features (i.e., repetitive patterns that are near each other) in the tested
    sequence that would indicate a deviation from the assumption of randomness. The intention is to detect whether
    the number of peaks exceeding the 95 % threshold is significantly different than 5 %.
    :param bin_data: a binary string
    :return: the p-value from the test
    """
    n = len(bin_data)
    plus_minus_one = []
    for char in tqdm(bin_data):
        if char == '0':
            plus_minus_one.append(-1)
        elif char == '1':
            plus_minus_one.append(1)
    # Product discrete fourier transform of plus minus one
    s = sff.fft(plus_minus_one)
    #print(s)
    modulus = numpy.abs(s[0: int(n / 2)])
    #print(modulus)
    tau = numpy.sqrt(math.log(1 / 0.05) * n)
    # Theoretical number of peaks
    count_n0 = 0.95 * (n / 2)
    # Count the number of actual peaks m > T
    count_n1 = len(numpy.where(modulus < tau)[0])
    # Calculate d and return the p value statistic
    d = (count_n1 - count_n0) / (numpy.sqrt((n * 0.95 * 0.05) / 4))
    result = spc.erfc(abs(d) / numpy.sqrt(2))
    if result >= 0.01:
        return f'------------ \nSpectral Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nSpectral Test \nUnsuccess P-value = {str(result)} \n------------'
