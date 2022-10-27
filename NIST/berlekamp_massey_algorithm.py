from numpy import zeros, dot
from copy import copy


def berlekamp_massey_algorithm(bin_data: list):
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
    n = len(bin_data)
    c = zeros(n)
    b = zeros(n)
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    while i < n:
        v = bin_data[(i - l):i]
        v = v[::-1]
        cc = c[1:l + 1]
        d = (bin_data[i] + dot(v, cc)) % 2
        if d == 1:
            temp = copy(c)
            p = zeros(n)
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