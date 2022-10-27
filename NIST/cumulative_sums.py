from tqdm import trange, tqdm
from numpy import zeros, max, floor, sqrt, array, abs, sum
import scipy.stats as sst

def cumulative_sums(bin_data: str, path: str, method="forward"):
    """
    Note that this description is taken from the NIST documentation [1]
    [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
    The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative sum of
    adjusted (-1, +1) digits in the sequence. The purpose of the test is to determine whether the cumulative sum of
    the partial sequences occurring in the tested sequence is too large or too small relative to the expected
    behavior of that cumulative sum for random sequences. This cumulative sum may be considered as a random walk.
    For a random sequence, the excursions of the random walk should be near zero. For certain types of non-random
    sequences, the excursions of this random walk from zero will be large.
    :param bin_data: a binary string
    :param method: the method used to calculate the statistic
    :return: the P-value
    """
    n = len(bin_data)
    counts = zeros(n)
    # Calculate the statistic using a walk forward
    if method != "forward":
        bin_data = bin_data[::-1]

    ix = 0
    for char in tqdm(bin_data):
        sub = 1
        if char == '0':
            sub = -1
        if ix > 0:
            counts[ix] = counts[ix - 1] + sub
        else:
            counts[ix] = sub
        ix += 1

    # This is the maximum absolute level obtained by the sequence
    abs_max = max(abs(counts))

    start = int(floor(0.25 * floor(-n / abs_max) + 1))
    end = int(floor(0.25 * floor(n / abs_max) - 1))
    terms_one = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k - 1) * abs_max / sqrt(n))
        terms_one.append(sst.norm.cdf((4 * k + 1) * abs_max / sqrt(n)) - sub)

    start = int(floor(0.25 * floor(-n / abs_max - 3)))
    end = int(floor(0.25 * floor(n / abs_max) - 1))
    terms_two = []
    for k in trange(start, end + 1):
        sub = sst.norm.cdf((4 * k + 1) * abs_max / sqrt(n))
        terms_two.append(sst.norm.cdf((4 * k + 3) * abs_max / sqrt(n)) - sub)

    result = 1.0 - sum(array(terms_one))
    result += sum(array(terms_two))
    if result >= 0.01:
        open(path, 'a').write(
            f'------------\nCumulative Sums Test\nSuccess P-value = {str(result)}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------\nCumulative Sums Test\nUnsuccess P-value = {str(result)}\n------------\n')

    return 0
