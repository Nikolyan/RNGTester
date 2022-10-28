from math import erfc

def frequency_monobit_test(bin_data: list, path: str):
    result = erfc((abs(bin_data.count(1) - bin_data.count(0)) / (len(bin_data) ** (1 / 2))) / 2 ** (1 / 2))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nFrequency Monobit Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nFrequency Monobit Test\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0