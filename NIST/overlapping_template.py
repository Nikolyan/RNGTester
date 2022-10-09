import math
from tqdm import trange, tqdm
import scipy.special as spc
import numpy

def get_prob(u, x):
    out = 1.0 * numpy.exp(-x)
    if u != 0:
        out = 1.0 * x * numpy.exp(2 * -x) * (2 ** -u) * spc.hyp1f1(u + 1, 2, x)
    return out


def overlapping_template(bin_data: str, template_size=9, block_size=1032):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of the Overlapping Template Matching test is the number of occurrences of pre-specified target
    strings. Both this test and the Non-overlapping Template Matching test of Section 2.7 use an m-bit
    window to search for a specific m-bit pattern. As with the test in Section 2.7, if the pattern is not found,
    the window slides one bit position. The difference between this test and the test in Section 2.7 is that
    when the pattern is found, the window slides only one bit before resuming the search.
    :param bin_data: a binary string
    :param pattern_size: the length of the pattern
    :return: the p-value from the test
    """
    n = len(bin_data)
    pattern = ""
    for i in range(template_size):
        pattern += "1"
    num_blocks = math.floor(n / block_size)
    lambda_val = float(block_size - template_size + 1) / pow(2, template_size)
    eta = lambda_val / 2.0

    piks = [get_prob(i, eta) for i in range(5)]
    diff = float(numpy.array(piks).sum())
    piks.append(1.0 - diff)

    pattern_counts = numpy.zeros(6)
    for i in trange(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block_data = bin_data[block_start:block_end]
        # Count the number of pattern hits
        pattern_count = 0
        j = 0
        while j < block_size:
            sub_block = block_data[j:j + template_size]
            if sub_block == pattern:
                pattern_count += 1
            j += 1
        if pattern_count <= 4:
            pattern_counts[pattern_count] += 1
        else:
            pattern_counts[5] += 1

    chi_squared = 0.0
    for i in trange(len(pattern_counts)):
        chi_squared += pow(pattern_counts[i] - num_blocks * piks[i], 2.0) / (num_blocks * piks[i])
    result = spc.gammaincc(5.0 / 2.0, chi_squared / 2.0)
    if result >= 0.01:
        return f'------------ \nOverlapping Patterns Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nOverlapping Patterns Test \nUnsuccess P-value = {str(result)} \n------------'
