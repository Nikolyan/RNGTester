from tqdm import tqdm
import scipy.special as spc
from NIST.berlekamp_massey_algorithm import berlekamp_massey_algorithm
from numpy import histogram


def linear_complexity(bin_data: list, path: str, block_size=500):
    dof = 6
    piks = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

    t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
    mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2
    number_of_blocks = int(len(bin_data) / block_size)
    if number_of_blocks >= 1:
        block_end = block_size
        block_start = 0
        array_of_blocks = []
        for i in range(number_of_blocks):
            array_of_blocks.append(bin_data[block_start:block_end])
            block_start += block_size
            block_end += block_size
        array_of_complex = []
        for block in tqdm(array_of_blocks):
            array_of_complex.append(berlekamp_massey_algorithm(block))
        t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in array_of_complex])
        vg = histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
        im = ([((vg[ii] - number_of_blocks * piks[ii]) ** 2) / (number_of_blocks * piks[ii]) for ii in range(7)])
        chi = 0.0
        for i in range(len(piks)):
            chi += im[i]
        result = spc.gammaincc(dof / 2.0, chi / 2.0)
        if result >= 0.01:
            open(path, 'a').write(
                f'------------\nLinear Complexity Test\nSuccess P-value = {str(result)}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------\nLinear Complexity Test\nUnsuccess P-value = {str(result)}\n------------\n')
        return 0
    else:
        return 'Error'
