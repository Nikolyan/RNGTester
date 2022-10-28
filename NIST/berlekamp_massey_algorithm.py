from numpy import zeros, dot
from copy import copy


def berlekamp_massey_algorithm(bin_data: list):
    n = len(bin_data)
    c_arr = zeros(n)
    b_arr = zeros(n)
    c_arr[0], b_arr[0] = 1, 1
    l, m, i = 0, -1, 0
    while i < n:
        v = bin_data[(i - l):i]
        v = v[::-1]
        cc = c_arr[1:l + 1]
        d = (bin_data[i] + dot(v, cc)) % 2
        if d == 1:
            new_c_arr = copy(c_arr)
            p = zeros(n)
            for j in range(0, l):
                if b_arr[j] == 1:
                    p[j + i - m] = 1
            c_arr = (c_arr + p) % 2
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b_arr = new_c_arr
        i += 1
    return l