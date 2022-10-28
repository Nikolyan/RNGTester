from math import floor, log, erfc, sqrt
from numpy import zeros
from tqdm import trange

def universal_statistical_test(bin_data: str, path: str):
    length_of_binary_data = len(bin_data)

    pattern_size = 5
    if length_of_binary_data >= 387840:
        pattern_size = 6
    elif length_of_binary_data >= 904960:
        pattern_size = 7
    elif length_of_binary_data >= 2068480:
        pattern_size = 8
    elif length_of_binary_data >= 4654080:
        pattern_size = 9
    elif length_of_binary_data >= 10342400:
        pattern_size = 10
    elif length_of_binary_data >= 22753280:
        pattern_size = 11
    elif length_of_binary_data >= 49643520:
        pattern_size = 12
    elif length_of_binary_data >= 107560960:
        pattern_size = 13
    elif length_of_binary_data >= 231669760:
        pattern_size = 14
    elif length_of_binary_data >= 496435200:
        pattern_size = 15
    elif length_of_binary_data >= 1059061760:
        pattern_size = 16
    else:
        print("Error, small size of array")


    if 5 < pattern_size < 16:
        ones = ""
        for i in range(pattern_size):
            ones += "1"

        num_ints = int(ones, 2)
        vobs = zeros(num_ints + 1)

        num_of_blocks = floor(length_of_binary_data / pattern_size)
        init_bits = 10 * pow(2, pattern_size)

        test_bits = num_of_blocks - init_bits
        c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
        variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
        expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
        sigma = c * sqrt(variance[pattern_size] / test_bits)

        sum = 0.0
        for i in trange(num_of_blocks):
            block_start = i * pattern_size
            block_end = block_start + pattern_size
            block_data = bin_data[block_start: block_end]
            rep = int(block_data, 2)
            if i < init_bits:
                vobs[rep] = i + 1
            else:
                init = vobs[rep]
                vobs[rep] = i + 1
                sum += log(i - init + 1, 2)

        phi = float(sum / test_bits)
        stat = abs(phi - expected[pattern_size]) / (float(sqrt(2)) * sigma)
        result = erfc(stat)

        if result >= 0.01:
            open(path, 'a').write(
                f'------------\nUniversal Statistical Test\nSuccess P-value = {str(result)}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------\nUniversal Statistical Test\nUnsuccess P-value = {str(result)}\n------------\n')

        return 0

    else:
        return 'Error'
