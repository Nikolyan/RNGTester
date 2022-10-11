from tqdm import trange
import scipy.special as spc
from numpy import zeros, ones, cumsum, abs, where, sqrt

def get_frequency(list_data, trigger):
    """
    This method is used by the random_excursions_variant method to get frequencies
    """
    frequency = 0
    for (x, y) in list_data:
        if x == trigger:
            frequency = y
    return frequency


def random_excursions_variant(bin_data):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the total number of times that a particular state is visited (i.e., occurs) in a
    cumulative sum random walk. The purpose of this test is to detect deviations from the expected number of visits
    to various states in the random walk. This test is actually a series of eighteen tests (and conclusions), one
    test and conclusion for each of the states: -9, -8, …, -1 and +1, +2, …, +9.
    :param bin_data: a binary string
    :return: the P-value
    """
    int_data = zeros(len(bin_data))
    for i in trange(len(bin_data)):
        int_data[i] = int(bin_data[i])
    sum_int = (2 * int_data) - ones(len(int_data))
    cumulative_sum = cumsum(sum_int)

    li_data = []
    for xs in sorted(set(cumulative_sum)):
        if abs(xs) <= 9:
            li_data.append([xs, len(where(cumulative_sum == xs)[0])])

    j = get_frequency(li_data, 0) + 1
    result = []
    for xs in range(-9, 9 + 1):
        if not xs == 0:
            den = sqrt(2 * j * (4 * abs(xs) - 2))
            result = spc.erfc(abs(get_frequency(li_data, xs) - j) / den)
            if result >= 0.01:
                print(
                    f'------------ \nRandom Excursions Variant Test {xs} \nSuccess P-value = {str(result)} \n------------')
            else:
                print(
                    f'------------ \nRandom Excursions Variant Test {xs} \nUnsuccess P-value = {str(result)} \n------------')