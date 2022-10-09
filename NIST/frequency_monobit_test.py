from math import erfc

def frequency_monobit_test(array: list):
    """The focus of the test is the proportion of zeroes and ones for the entire sequence. The purpose of this test
    is to determine whether the number of ones and zeros in a sequence are approximately the same as would be
    expected for a truly random sequence. The test assesses the closeness of the fraction of ones to 1⁄2, that is,
    the number of ones and zeroes in a sequence should be about the same. All subsequent tests depend on the passing
    of this test.
    """

    result = erfc((abs(array.count(1) - array.count(0)) / (len(array) ** (1 / 2))) / 2 ** (1 / 2))
    if result >= 0.01:
        return f'------------ \nFrequency Monobit Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nFrequency Monobit Test \nUnsuccess P-value = {str(result)} \n------------'
