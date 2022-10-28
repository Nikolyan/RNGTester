from tqdm import trange, tqdm
from numpy import zeros, max, floor, sqrt, array, abs, sum
import scipy.stats as sst

def cumulative_sums(bin_data: str, path: str, method="forward"):
    n = len(bin_data)
    counters = zeros(n)
    if method != "forward":
        bin_data = bin_data[::-1]

    ix = 0
    for char in tqdm(bin_data):
        sub = 1
        if char == '0':
            sub = -1
        if ix > 0:
            counters[ix] = counters[ix - 1] + sub
        else:
            counters[ix] = sub
        ix += 1

    abs_max = max(abs(counters))

    start = int(floor(0.25 * floor(-n / abs_max) + 1))
    end = int(floor(0.25 * floor(n / abs_max) - 1))
    one_terms = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k - 1) * abs_max / sqrt(n))
        one_terms.append(sst.norm.cdf((4 * k + 1) * abs_max / sqrt(n)) - sub)

    start = int(floor(0.25 * floor(-n / abs_max - 3)))
    end = int(floor(0.25 * floor(n / abs_max) - 1))
    two_terms = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k + 1) * abs_max / sqrt(n))
        two_terms.append(sst.norm.cdf((4 * k + 3) * abs_max / sqrt(n)) - sub)

    result = 1.0 - sum(array(one_terms))
    result += sum(array(two_terms))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nCumulative Sums Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nCumulative Sums Test\nUnsuccess P-value = {str(result)}\n------------\n')

    return 0
