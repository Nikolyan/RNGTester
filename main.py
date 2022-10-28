from creat_array import creat_array
from NIST import *
from Methods import *
import time

path = r'.\nist.txt'
m = r'.\results.txt'

bin = creat_array(r'.\dt.bin', 0)

independence_of_signs(bin, m)

# string = ''.join([str(x) for x in bin])
# print(frequency_monobit_test(bin, path))
# print(frequency_block_test(bin, path, 128))
# print(runs(bin, path))
# print(longest_runs(bin, path))
# print(matrix_rank(bin, path))
# print(spectral_test(bin, path))
# print(non_overlapping_template_test(bin, path))
# print(overlapping_template(bin, path))###???
# print(universal_statistical_test(string, path))
# print(linear_complexity(bin, path))
# print(serial(string, path))
# print(approximate_entropy(string, path))
# print(cumulative_sums(bin, path))
# print(random_excursions_variant(bin, path))
