from math import erfc
from tqdm import trange


def runs(bin_data: list, path: str):

    n = len(bin_data)
    pi = sum(bin_data) / n
    obs_array = []
    for j in trange(0, n - 1):
        if bin_data[j] == bin_data[j + 1]:
            obs_array.append(0)
        else:
            obs_array.append(1)
    obs = sum(obs_array) + 1
    result = erfc(abs(obs - 2 * n * pi * (1 - pi)) / (2 * (2 * n) ** (1 / 2) * pi * (1 - pi)))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nRuns Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nRuns Test\nUnsuccess P-value = {str(result)}\n------------\n')
    return 0