from math import log, floor
import scipy.special as spc
from numpy import zeros
from tqdm import trange

def non_overlapping_template_test(bin_data: str, template="000000001", num_blocks=8):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the number of occurrences of pre-specified target strings. The purpose of this
    test is to detect generators that produce too many occurrences of a given non-periodic (aperiodic) pattern.
    For this test and for the Overlapping Template Matching test of Section 2.8, an m-bit window is used to
    search for a specific m-bit pattern. If the pattern is not found, the window slides one bit position. If the
    pattern is found, the window is reset to the bit after the found pattern, and the search resumes.
    :param bin_data: a binary string
    :param pattern: the pattern to match to
    :return: the p-value from the test
    """
    n = len(bin_data)
    pattern_size = len(template)
    block_size = floor(n / num_blocks)
    pattern_counts = zeros(num_blocks)
    # For each block in the data
    for i in trange(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block_data = bin_data[block_start:block_end]
        # Count the number of pattern hits
        j = 0
        while j < block_size:
            sub_block = block_data[j:j + pattern_size]
            if sub_block == template:
                pattern_counts[i] += 1
                j += pattern_size
            else:
                j += 1
    # Calculate the theoretical mean and variance
    mean = (block_size - pattern_size + 1) / pow(2, pattern_size)
    var = block_size * ((1 / pow(2, pattern_size)) - (((2 * pattern_size) - 1) / (pow(2, pattern_size * 2))))
    # Calculate the Chi Squared statistic for these pattern matches
    chi_squared = 0
    for i in trange(num_blocks):
        chi_squared += pow(pattern_counts[i] - mean, 2.0) / var
    # Calculate and return the p value statistic
    result = spc.gammaincc(num_blocks / 2, chi_squared / 2)
    if result >= 0.01:
        return f'------------ \nNon Overlapping Template Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nNon Overlapping Template Test \nUnsuccess P-value = {str(result)} \n------------'
