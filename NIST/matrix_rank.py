from math import exp, floor
from numpy import zeros
from tqdm import trange
from NIST.Bin_matrix import Bin_matrix

def matrix_rank(bin_data: list, path: str, q=32):
    shape = (q, q)
    n = len(bin_data)
    size_of_block = q * q
    num = floor(n / (q * q))
    block_start, block_end = 0, size_of_block
    if num > 0:
        max_ranks = [0, 0, 0]
        for j in trange(num):
            data_block = bin_data[block_start:block_end]
            block = zeros(len(data_block))
            for i in range(len(data_block)):
                if data_block[i] == 1:
                    block[i] = 1.0
            m = block.reshape(shape)
            rank = Bin_matrix(m, q, q).compute_rank()
            if rank == q:
                max_ranks[0] += 1
            elif rank == (q - 1):
                max_ranks[1] += 1
            else:
                max_ranks[2] += 1
            block_start += size_of_block
            block_end += size_of_block
        piks = [1.0, 0.0, 0.0]
        for x in range(1, 50):
            piks[0] *= 1 - (1.0 / (2 ** x))
        piks[1] = 2 * piks[0]
        piks[2] = 1 - piks[0] - piks[1]

        chi = 0.0
        for i in range(len(piks)):
            chi += ((max_ranks[i] - piks[i] * num)**2.0) / (piks[i] * num)
        result = exp(-chi / 2)
        if result >= 0.01:
            open(path, 'a').write(
                f'------------\nMatrix Rank Test\nSuccess P-value = {str(result)}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------\nMatrix Rank Test\nUnsuccess P-value = {str(result)}\n------------\n')
        return 0
    else:
        return 'Error'
