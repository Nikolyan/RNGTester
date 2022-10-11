from math import exp, floor
from numpy import zeros
from tqdm import trange
from NIST.BinaryMatrix import BinaryMatrix

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
    num_m = floor(n / (q * q))
    block_start, block_end = 0, block_size
    # print(q, n, num_m, block_size)

    if num_m > 0:
        max_ranks = [0, 0, 0]
        for im in trange(num_m):
            block_data = bin_data[block_start:block_end]
            block = zeros(len(block_data))
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
        result = exp(-chi / 2)
        if result >= 0.01:
            return f'------------ \nMatrix Rank Test \nSuccess P-value = {str(result)} \n------------'
        else:
            return f'------------ \nMatrix Rank Test \nUnsuccess P-value = {str(result)} \n------------'
    else:
        return 'Error'