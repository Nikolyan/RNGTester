import math
import copy
from tqdm import trange, tqdm
import scipy.special as spc

def runs(array: list):

    new_array = copy.deepcopy(array)
    n = len(new_array)
    pi = sum(new_array) / n
    obs_array = []
    for j in trange(0, n - 1):
        if new_array[j] == new_array[j + 1]:
            obs_array.append(0)
        else:
            obs_array.append(1)
    obs = sum(obs_array) + 1

    result = math.erfc(abs(obs - 2 * n * pi * (1 - pi)) / (2 * (2 * n) ** (1 / 2) * pi * (1 - pi)))

    if result >= 0.01:
        return f'------------ \nRuns Test \nSuccess P-value = {str(result)} \n------------'
    else:
        return f'------------ \nRuns Test \nUnsuccess P-value = {str(result)} \n------------'
