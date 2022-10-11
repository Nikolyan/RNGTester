from tqdm import tqdm
import scipy.special as spc
from NIST.berlekamp_massey_algorithm import berlekamp_massey_algorithm
from numpy import histogram

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

    if num_blocks >= 1:
        block_end = block_size
        block_start = 0
        blocks = []
        for i in range(num_blocks):


            blocks.append(bin_data[block_start:block_end])
            block_start += block_size
            block_end += block_size

        complexities = []
        for block in tqdm(blocks):

            complexities.append(berlekamp_massey_algorithm(block))

        t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])
        vg = histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
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