import copy
from tqdm import tqdm, trange
import math
from math import floor, log, sqrt


from numpy import zeros
import numpy.fft as sff

import scipy.stats as sst
import scipy.special as spc
from scipy.special import erfc


import numpy


def frequency_monobit_test(array: list):
    """The focus of the test is the proportion of zeroes and ones for the entire sequence. The purpose of this test
    is to determine whether the number of ones and zeros in a sequence are approximately the same as would be
    expected for a truly random sequence. The test assesses the closeness of the fraction of ones to 1⁄2, that is,
    the number of ones and zeroes in a sequence should be about the same. All subsequent tests depend on the passing
    of this test.
    """

    result = math.erfc((abs(array.count(1) - array.count(0)) / (len(array) ** (1 / 2))) / 2 ** (1 / 2))
    if result >= 0.01:
        return f'------------ \nFrequency Monobit Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nFrequency Monobit Test \nUnsuccess P-value = {str(result)} \n------------'


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


def runs(array: list):

    new_array = copy.deepcopy(array)
    n = len(new_array)
    pi = sum(new_array) / n
    print(n)
    print(2 / n ** (1 / 2))
    # if abs(pi - 0.5) >= t:
    #     print('Runs Test can not done, because |p - 0.5| >= t')
    #     print('pi = ', pi, 't = ', t)
    #     print('p = 0.000')
    #     print('------------')
    #     return 0.000
    # else:
    obs_array = []
    for j in trange(0, n - 1):
        if new_array[j] == new_array[j + 1]:
            obs_array.append(0)
        else:
            obs_array.append(1)
    obs = sum(obs_array) + 1

    result = math.erfc(abs(obs - 2 * n * pi * (1 - pi)) / (2 * (2 * n) ** (1 / 2) * pi * (1 - pi)))

    if result >= 0.01:
        return f'------------ \nRuns Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nRuns Test \nUnsuccess P-value = {str(result)} \n------------'


def longest_runs(array):
    bin_data = copy.deepcopy(array)

    if len(bin_data) < 128:
        print("\t", "Not enough data to run test!")
        return -1.0
    elif len(bin_data) < 6272:
        k, m = 3, 8
        v_values = [1, 2, 3, 4]
        pik_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif len(bin_data) < 75000:
        k, m = 5, 128
        v_values = [4, 5, 6, 7, 8, 9]
        pik_values = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        k, m = 6, 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pik_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    # Work out the number of blocks, discard the remainder
    # pik = [0.2148, 0.3672, 0.2305, 0.1875]
    num_blocks = math.floor(len(bin_data) / m)
    frequencies = numpy.zeros(k + 1)
    block_start, block_end = 0, m
    for i in trange(num_blocks):
        # Slice the binary string into a block
        block_data = bin_data[block_start:block_end]
        # Keep track of the number of ones
        max_run_count, run_count = 0, 0
        for j in range(0, m):
            if block_data[j] == '1':
                run_count += 1
                max_run_count = max(max_run_count, run_count)
            else:
                max_run_count = max(max_run_count, run_count)
                run_count = 0
        max_run_count = max(max_run_count, run_count)
        if max_run_count < v_values[0]:
            frequencies[0] += 1
        for j in range(k):
            if max_run_count == v_values[j]:
                frequencies[j] += 1
        if max_run_count > v_values[k - 1]:
            frequencies[k] += 1
        block_start += m
        block_end += m
    chi_squared = 0
    for i in range(len(frequencies)):
        chi_squared += (pow(frequencies[i] - (num_blocks * pik_values[i]), 2.0)) / (num_blocks * pik_values[i])
    result = spc.gammaincc(float(k / 2), float(chi_squared / 2))

    if result >= 0.01:
        return f'------------ \nLongest Runs Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nLongest Runs Test \nUnsuccess P-value = {str(result)} \n------------'


class BinaryMatrix:
    def __init__(self, matrix, rows, cols):
        """
        This class contains the algorithm specified in the NIST suite for computing the **binary rank** of a matrix.
        :param matrix: the matrix we want to compute the rank for
        :param rows: the number of rows
        :param cols: the number of columns
        :return: a BinaryMatrix object
        """
        self.M = rows
        self.Q = cols
        self.A = matrix
        self.L = min(rows, cols)

    def compute_rank(self, verbose=False):
        """
        This method computes the binary rank of self.matrix
        :param verbose: if this is true it prints out the matrix after the forward elimination and backward elimination
        operations on the rows. This was used to testing the method to check it is working as expected.
        :return: the rank of the matrix.
        """
        if verbose:
            print("Original Matrix\n", self.A)

        i = 0
        while i < self.L - 1:
            if self.A[i][i] == 1:
                self.perform_row_operations(i, True)
            else:
                found = self.find_unit_element_swap(i, True)
                if found == 1:
                    self.perform_row_operations(i, True)
            i += 1

        if verbose:
            print("Intermediate Matrix\n", self.A)

        i = self.M - 1
        while i > 0:
            if self.A[i][i] == 1:
                self.perform_row_operations(i, False)
            else:
                if self.find_unit_element_swap(i, False) == 1:
                    self.perform_row_operations(i, False)
            i -= 1

        if verbose:
            print("Final Matrix\n", self.A)

        return self.determine_rank()

    def perform_row_operations(self, i, forward_elimination):
        """
        This method performs the elementary row operations. This involves xor'ing up to two rows together depending on
        whether or not certain elements in the matrix contain 1's if the "current" element does not.
        :param i: the current index we are are looking at
        :param forward_elimination: True or False.
        """
        if forward_elimination:
            j = i + 1
            while j < self.M:
                if self.A[j][i] == 1:
                    self.A[j, :] = (self.A[j, :] + self.A[i, :]) % 2
                j += 1
        else:
            j = i - 1
            while j >= 0:
                if self.A[j][i] == 1:
                    self.A[j, :] = (self.A[j, :] + self.A[i, :]) % 2
                j -= 1

    def find_unit_element_swap(self, i, forward_elimination):
        """
        This given an index which does not contain a 1 this searches through the rows below the index to see which rows
        contain 1's, if they do then they swapped. This is done on the forward and backward elimination
        :param i: the current index we are looking at
        :param forward_elimination: True or False.
        """
        row_op = 0
        if forward_elimination:
            index = i + 1
            while index < self.M and self.A[index][i] == 0:
                index += 1
            if index < self.M:
                row_op = self.swap_rows(i, index)
        else:
            index = i - 1
            while index >= 0 and self.A[index][i] == 0:
                index -= 1
            if index >= 0:
                row_op = self.swap_rows(i, index)
        return row_op

    def swap_rows(self, i, ix):
        """
        This method just swaps two rows in a matrix. Had to use the copy package to ensure no memory leakage
        :param i: the first row we want to swap and
        :param ix: the row we want to swap it with
        :return: 1
        """
        temp = copy.copy(self.A[i, :])
        self.A[i, :] = self.A[ix, :]
        self.A[ix, :] = temp
        return 1

    def determine_rank(self):
        """
        This method determines the rank of the transformed matrix
        :return: the rank of the transformed matrix
        """
        rank = self.M
        i = 0
        while i < self.M:
            all_zeros = 1
            for j in range(self.Q):
                if self.A[i][j] == 1:
                    all_zeros = 0
            if all_zeros == 1:
                rank -= 1
            i += 1
        return rank


def matrix_rank(bin_data: str, q=32):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of the test is the rank of disjoint sub-matrices of the entire sequence. The purpose of this test is
    to check for linear dependence among fixed length sub strings of the original sequence. Note that this test
    also appears in the DIEHARD battery of tests.
    :param bin_data: a binary string
    :return: the p-value from the test
    """


    shape = (q, q)
    n = len(bin_data)
    block_size = int(q * q)
    num_m = math.floor(n / (q * q))
    block_start, block_end = 0, block_size
    # print(q, n, num_m, block_size)

    if num_m > 0:
        max_ranks = [0, 0, 0]
        for im in trange(num_m):
            block_data = bin_data[block_start:block_end]
            block = numpy.zeros(len(block_data))
            for i in range(len(block_data)):
                if block_data[i] == '1':
                    block[i] = 1.0
            m = block.reshape(shape)
            ranker = BinaryMatrix(m, q, q)
            rank = ranker.compute_rank()
            # print(rank)
            if rank == q:
                max_ranks[0] += 1
            elif rank == (q - 1):
                max_ranks[1] += 1
            else:
                max_ranks[2] += 1
            # Update index trackers
            block_start += block_size
            block_end += block_size

        piks = [1.0, 0.0, 0.0]
        for x in range(1, 50):
            piks[0] *= 1 - (1.0 / (2 ** x))
        piks[1] = 2 * piks[0]
        piks[2] = 1 - piks[0] - piks[1]

        chi = 0.0
        for i in range(len(piks)):
            chi += pow((max_ranks[i] - piks[i] * num_m), 2.0) / (piks[i] * num_m)
        result = math.exp(-chi / 2)
        if result >= 0.01:
            return f'------------ \nMatrix Rank Test \nSuccess P-value = {str(result)} \n------------'
        else:
            return f'------------ \nMatrix Rank Test \nUnsuccess P-value = {str(result)} \n------------'
    else:
        return 'Error'


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
    block_size = math.floor(n / num_blocks)
    pattern_counts = numpy.zeros(num_blocks)
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


def universal_statistical_test(binary_data: str):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the number of bits between matching patterns (a measure that is related to the
    length of a compressed sequence). The purpose of the test is to detect whether or not the sequence can be
    significantly compressed without loss of information. A significantly compressible sequence is considered
    to be non-random. **This test is always skipped because the requirements on the lengths of the binary
    strings are too high i.e. there have not been enough trading days to meet the requirements.
    :param      binary_data:    a binary string
    :param      verbose         True to display the debug messgae, False to turn off debug message
    :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)
    """
    length_of_binary_data = len(binary_data)
    pattern_size = 5
    if length_of_binary_data >= 387840:
        pattern_size = 6
    if length_of_binary_data >= 904960:
        pattern_size = 7
    if length_of_binary_data >= 2068480:
        pattern_size = 8
    if length_of_binary_data >= 4654080:
        pattern_size = 9
    if length_of_binary_data >= 10342400:
        pattern_size = 10
    if length_of_binary_data >= 22753280:
        pattern_size = 11
    if length_of_binary_data >= 49643520:
        pattern_size = 12
    if length_of_binary_data >= 107560960:
        pattern_size = 13
    if length_of_binary_data >= 231669760:
        pattern_size = 14
    if length_of_binary_data >= 496435200:
        pattern_size = 15
    if length_of_binary_data >= 1059061760:
        pattern_size = 16

    if 5 < pattern_size < 16:
        # Create the biggest binary string of length pattern_size
        ones = ""
        for i in range(pattern_size):
            ones += "1"

        # How long the state list should be
        num_ints = int(ones, 2)
        vobs = zeros(num_ints + 1)

        # Keeps track of the blocks, and whether were are initializing or summing
        num_blocks = floor(length_of_binary_data / pattern_size)
        #Q = 10 * pow(2, pattern_size)
        init_bits = 10 * pow(2, pattern_size)

        test_bits = num_blocks - init_bits

        # These are the expected values assuming randomness (uniform)
        c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
        variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
        expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                    10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
        sigma = c * sqrt(variance[pattern_size] / test_bits)

        cumsum = 0.0
        # Examine each of the K blocks in the test segment and determine the number of blocks since the
        # last occurrence of the same L-bit block (i.e., i – Tj). Replace the value in the table with the
        # location of the current block (i.e., Tj= i). Add the calculated distance between re-occurrences of
        # the same L-bit block to an accumulating log2 sum of all the differences detected in the K blocks
        for i in trange(num_blocks):
            block_start = i * pattern_size
            block_end = block_start + pattern_size
            block_data = binary_data[block_start: block_end]
            # Work out what state we are in
            int_rep = int(block_data, 2)

            # Initialize the state list
            if i < init_bits:
                vobs[int_rep] = i + 1
            else:
                initial = vobs[int_rep]
                vobs[int_rep] = i + 1
                cumsum += log(i - initial + 1, 2)

        # Compute the statistic
        phi = float(cumsum / test_bits)
        stat = abs(phi - expected[pattern_size]) / (float(sqrt(2)) * sigma)

        # Compute for P-Value
        result = erfc(stat)

        if result >= 0.01:
            return f'------------ \nUniversal Statistical Test \nSuccess P-value = {str(result)} \n------------'
        else:
            return f'------------ \nUniversal Statistical Test \nUnsuccess P-value = {str(result)} \n------------'
    else:
        return 'Error'


def berlekamp_massey_algorithm(block_data):
    """
    An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
    [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
    The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
    for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
    sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
    non-zero elements to have a multiplicative inverse.
    :param block_data:
    :return:
    """
    n = len(block_data)
    c = numpy.zeros(n)
    b = numpy.zeros(n)
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    int_data = [int(el) for el in block_data]
    while tqdm(i < n):
        v = int_data[(i - l):i]
        v = v[::-1]
        cc = c[1:l + 1]
        d = (int_data[i] + numpy.dot(v, cc)) % 2
        if d == 1:
            temp = copy.copy(c)
            p = numpy.zeros(n)
            for j in range(0, l):
                if b[j] == 1:
                    p[j + i - m] = 1
            c = (c + p) % 2
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b = temp
        i += 1
    return l


def linear_complexity(bin_data, block_size=500):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the length of a linear feedback shift register (LFSR). The purpose of this test is to
    determine whether or not the sequence is complex enough to be considered random. Random sequences are
    characterized by longer LFSRs. An LFSR that is too short implies non-randomness.
    :param bin_data: a binary string
    :param block_size: the size of the blocks to divide bin_data into. Recommended block_size >= 500
    :return:
    """
    dof = 6
    piks = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

    t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
    mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2

    num_blocks = int(len(bin_data) / block_size)
    if num_blocks > 1:
        block_end = block_size
        block_start = 0
        blocks = []
        for i in trange(num_blocks):
            blocks.append(bin_data[block_start:block_end])
            block_start += block_size
            block_end += block_size

        complexities = []
        for block in blocks:
            complexities.append(berlekamp_massey_algorithm(block))

        t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])
        vg = numpy.histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
        im = ([((vg[ii] - num_blocks * piks[ii]) ** 2) / (num_blocks * piks[ii]) for ii in range(7)])

        chi_squared = 0.0
        for i in range(len(piks)):
            chi_squared += im[i]
        result = spc.gammaincc(dof / 2.0, chi_squared / 2.0)
        if result >= 0.01:
            return f'------------ \nLinear Complexity Test \nSuccess P-value = {str(result)} \n------------'
        else:
            return f'------------ \nLinear Complexity Test \nUnsuccess P-value = {str(result)} \n------------'
    else:
        return 'Error'


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
    vobs_one = numpy.zeros(int(max_pattern[0:template_length:], 2) + 1)
    vobs_two = numpy.zeros(int(max_pattern[0:template_length - 1:], 2) + 1)
    vobs_thr = numpy.zeros(int(max_pattern[0:template_length - 2:], 2) + 1)

    for i in trange(n):
        # Work out what pattern is observed
        vobs_one[int(bin_data[i:i + template_length:], 2)] += 1
        vobs_two[int(bin_data[i:i + template_length - 1:], 2)] += 1
        vobs_thr[int(bin_data[i:i + template_length - 2:], 2)] += 1

    vobs = [vobs_one, vobs_two, vobs_thr]
    sums = numpy.zeros(3)
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


def approximate_entropy(bin_data: str, pattern_length=10):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    As with the Serial test of Section 2.11, the focus of this test is the frequency of all possible overlapping
    m-bit patterns across the entire sequence. The purpose of the test is to compare the frequency of overlapping
    blocks of two consecutive/adjacent lengths (m and m+1) against the expected result for a random sequence.
    :param bin_data: a binary string
    :param pattern_length: the length of the pattern (m)
    :return: the P value
    """
    n = len(bin_data)
    # Add first m+1 bits to the end
    # NOTE: documentation says m-1 bits but that doesnt make sense, or work.
    bin_data += bin_data[:pattern_length + 1:]

    # Get max length one patterns for m, m-1, m-2
    max_pattern = ''
    for i in range(pattern_length + 2):
        max_pattern += '1'

    # Keep track of each pattern's frequency (how often it appears)
    vobs_one = numpy.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs_two = numpy.zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

    for i in trange(n):
        # Work out what pattern is observed
        vobs_one[int(bin_data[i:i + pattern_length:], 2)] += 1
        vobs_two[int(bin_data[i:i + pattern_length + 1:], 2)] += 1

    # Calculate the test statistics and p values
    vobs = [vobs_one, vobs_two]
    sums = numpy.zeros(2)
    for i in range(2):
        for j in range(len(vobs[i])):
            if vobs[i][j] > 0:
                sums[i] += vobs[i][j] * math.log(vobs[i][j] / n)
    sums /= n
    ape = sums[0] - sums[1]
    chi_squared = 2.0 * n * (math.log(2) - ape)
    result = spc.gammaincc(pow(2, pattern_length - 1), chi_squared / 2.0)
    if result >= 0.01:
        return f'------------ \nApproximate Entropy Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nApproximate Entropy Test \nUnsuccess P-value = {str(result)} \n------------'


def cumulative_sums(bin_data: str, method="forward"):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative sum of
    adjusted (-1, +1) digits in the sequence. The purpose of the test is to determine whether the cumulative sum of
    the partial sequences occurring in the tested sequence is too large or too small relative to the expected
    behavior of that cumulative sum for random sequences. This cumulative sum may be considered as a random walk.
    For a random sequence, the excursions of the random walk should be near zero. For certain types of non-random
    sequences, the excursions of this random walk from zero will be large.
    :param bin_data: a binary string
    :param method: the method used to calculate the statistic
    :return: the P-value
    """
    n = len(bin_data)
    counts = numpy.zeros(n)
    # Calculate the statistic using a walk forward
    if method != "forward":
        bin_data = bin_data[::-1]

    ix = 0
    for char in tqdm(bin_data):
        sub = 1
        if char == '0':
            sub = -1
        if ix > 0:
            counts[ix] = counts[ix - 1] + sub
        else:
            counts[ix] = sub
        ix += 1

    # This is the maximum absolute level obtained by the sequence
    abs_max = numpy.max(numpy.abs(counts))

    start = int(numpy.floor(0.25 * numpy.floor(-n / abs_max) + 1))
    end = int(numpy.floor(0.25 * numpy.floor(n / abs_max) - 1))
    terms_one = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k - 1) * abs_max / numpy.sqrt(n))
        terms_one.append(sst.norm.cdf((4 * k + 1) * abs_max / numpy.sqrt(n)) - sub)

    start = int(numpy.floor(0.25 * numpy.floor(-n / abs_max - 3)))
    end = int(numpy.floor(0.25 * numpy.floor(n / abs_max) - 1))
    terms_two = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k + 1) * abs_max / numpy.sqrt(n))
        terms_two.append(sst.norm.cdf((4 * k + 3) * abs_max / numpy.sqrt(n)) - sub)

    result = 1.0 - numpy.sum(numpy.array(terms_one))
    result += numpy.sum(numpy.array(terms_two))
    if result >= 0.01:
        return f'------------ \nCumulative Sums Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nCumulative Sums Test \nUnsuccess P-value = {str(result)} \n------------'


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
    int_data = numpy.zeros(len(bin_data))
    for i in trange(len(bin_data)):
        int_data[i] = int(bin_data[i])
    sum_int = (2 * int_data) - numpy.ones(len(int_data))
    cumulative_sum = numpy.cumsum(sum_int)

    li_data = []
    for xs in sorted(set(cumulative_sum)):
        if numpy.abs(xs) <= 9:
            li_data.append([xs, len(numpy.where(cumulative_sum == xs)[0])])

    j = get_frequency(li_data, 0) + 1
    result = []
    for xs in range(-9, 9 + 1):
        if not xs == 0:
            den = numpy.sqrt(2 * j * (4 * numpy.abs(xs) - 2))
            result = spc.erfc(numpy.abs(get_frequency(li_data, xs) - j) / den)
            if result >= 0.01:
                print(
                    f'------------ \nRandom Excursions Variant Test {xs} \nSuccess P-value = {str(result)} \n------------')
            else:
                print(
                    f'------------ \nRandom Excursions Variant Test {xs} \nUnsuccess P-value = {str(result)} \n------------')
