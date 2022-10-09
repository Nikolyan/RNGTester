import math
from tqdm import trange, tqdm
import scipy.special as spc
import numpy


def get_pik_value(k, x):
    """
    This method is used by the random_excursions method to get expected probabilities
    """
    if k == 0:
        out = 1 - 1.0 / (2 * numpy.abs(x))
    elif k >= 5:
        out = (1.0 / (2 * numpy.abs(x))) * (1 - 1.0 / (2 * numpy.abs(x))) ** 4
    else:
        out = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * numpy.abs(x))) ** (k - 1)
    return out


def random_excursions(bin_data):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the number of cycles having exactly K visits in a cumulative sum random walk. The
    cumulative sum random walk is derived from partial sums after the (0,1) sequence is transferred to the
    appropriate (-1, +1) sequence. A cycle of a random walk consists of a sequence of steps of unit length taken at
    random that begin at and return to the origin. The purpose of this test is to determine if the number of visits
    to a particular state within a cycle deviates from what one would expect for a random sequence. This test is
    actually a series of eight tests (and conclusions), one test and conclusion for each of the states:
    States -> -4, -3, -2, -1 and +1, +2, +3, +4.
    :param bin_data: a binary string
    :return: the P-value
    """
    # Turn all the binary digits into +1 or -1
    int_data = numpy.zeros(len(bin_data))
    for i in trange(len(bin_data)):
        if bin_data[i] == '0':
            int_data[i] = -1.0
        else:
            int_data[i] = 1.0

    # Calculate the cumulative sum
    cumulative_sum = numpy.cumsum(int_data)
    # Append a 0 to the end and beginning of the sum
    cumulative_sum = numpy.append(cumulative_sum, [0])
    cumulative_sum = numpy.append([0], cumulative_sum)

    # These are the states we are going to look at
    x_values = numpy.array([-4, -3, -2, -1, 1, 2, 3, 4])

    # Identify all the locations where the cumulative sum revisits 0
    position = numpy.where(cumulative_sum == 0)[0]
    # For this identify all the cycles
    cycles = []
    for pos in range(len(position) - 1):
        # Add this cycle to the list of cycles
        cycles.append(cumulative_sum[position[pos]:position[pos + 1] + 1])
    num_cycles = len(cycles)

    state_count = []
    for cycle in cycles:
        # Determine the number of times each cycle visits each state
        state_count.append(([len(numpy.where(cycle == state)[0]) for state in x_values]))
    state_count = numpy.transpose(numpy.clip(state_count, 0, 5))

    su = []
    for cycle in range(6):
        su.append([(sct == cycle).sum() for sct in state_count])
    su = numpy.transpose(su)

    piks = ([([get_pik_value(uu, state) for uu in range(6)]) for state in x_values])
    inner_term = num_cycles * numpy.array(piks)
    chi = numpy.sum(1.0 * (numpy.array(su) - inner_term) ** 2 / inner_term, axis=1)
    result = ([spc.gammaincc(2.5, cs / 2.0) for cs in chi])
    if result >= 0.01:
        return f'------------ \nRandom Excursions Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nRandom Excursions Test \nUnsuccess P-value = {str(result)} \n------------'
