from math import erfc

def frequency_monobit_test(bin_data: list, path: str):
    """The focus of the test is the proportion of zeroes and ones for the entire sequence. The purpose of this test
    is to determine whether the number of ones and zeros in a sequence are approximately the same as would be
    expected for a truly random sequence. The test assesses the closeness of the fraction of ones to 1⁄2, that is,
    the number of ones and zeroes in a sequence should be about the same. All subsequent tests depend on the passing
    of this test.
    """

    result = erfc((abs(bin_data.count(1) - bin_data.count(0)) / (len(bin_data) ** (1 / 2))) / 2 ** (1 / 2))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nFrequency Monobit Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nFrequency Monobit Test\nUnsuccess P-value = {str(result)}\n------------\n')

    return 0
