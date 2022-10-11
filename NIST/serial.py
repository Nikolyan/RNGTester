from tqdm import trange
import scipy.special as spc
from numpy import zeros

def serial(bin_data, template_length=16):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the frequency of all possible overlapping m-bit patterns across the entire
    sequence. The purpose of this test is to determine whether the number of occurrences of the 2m m-bit
    overlapping patterns is approximately the same as would be expected for a random sequence. Random
    sequences have uniformity; that is, every m-bit pattern has the same chance of appearing as every other
    m-bit pattern. Note that for m = 1, the Serial test is equivalent to the Frequency test of Section 2.1.
    :param bin_data: a binary string
    :param pattern_length: the length of the pattern (m)
    :return: the P value
    """
    n = len(bin_data)
    # Add first m-1 bits to the end
    bin_data += bin_data[:template_length - 1:]

    # Get max length one patterns for m, m-1, m-2
    max_pattern = ''
    for i in range(template_length + 1):
        max_pattern += '1'

    # Keep track of each pattern's frequency (how often it appears)
    vobs_one = zeros(int(max_pattern[0:template_length:], 2) + 1)
    vobs_two = zeros(int(max_pattern[0:template_length - 1:], 2) + 1)
    vobs_thr = zeros(int(max_pattern[0:template_length - 2:], 2) + 1)

    for i in trange(n):
        # Work out what pattern is observed
        vobs_one[int(bin_data[i:i + template_length:], 2)] += 1
        vobs_two[int(bin_data[i:i + template_length - 1:], 2)] += 1
        vobs_thr[int(bin_data[i:i + template_length - 2:], 2)] += 1

    vobs = [vobs_one, vobs_two, vobs_thr]
    sums = zeros(3)
    for i in range(3):
        for j in range(len(vobs[i])):
            sums[i] += pow(vobs[i][j], 2)
        sums[i] = (sums[i] * pow(2, template_length - i) / n) - n

    # Calculate the test statistics and p values
    del1 = sums[0] - sums[1]
    del2 = sums[0] - 2.0 * sums[1] + sums[2]
    result = spc.gammaincc(pow(2, template_length - 1) / 2, del1 / 2.0)
    if result >= 0.01:
        return f'------------ \nSerial Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nSerial Test \nUnsuccess P-value = {str(result)} \n------------'