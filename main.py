from creat_array import creat_array
from NIST import *



ar = creat_array('dt.bin', 0)
print(frequency_monobit_test(ar))
print(frequency_block_test(ar, 128))
print(runs(ar))
print(longest_runs(ar))


string = ''.join([str(x) for x in ar])

print(matrix_rank(string))
print(spectral_test(string))
print(non_overlapping_template_test(string))
print(overlapping_template(string))
print(universal_statistical_test(string))
print(linear_complexity(string))
print(serial(string))
print(approximate_entropy(string))
print(cumulative_sums(string))
print(random_excursions_variant(string))
