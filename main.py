from creat_array import creat_array
import NIST



ar = creat_array(r'dt.bin', 1000)
print(NIST.frequency_monobit_test(ar))
print(NIST.frequency_block_test(ar, 128))
print(NIST.runs(ar))
print(NIST.longest_runs(ar))
string = ''.join([str(x) for x in ar])

print(NIST.matrix_rank(string))
print(NIST.spectral_test(string))
print(NIST.non_overlapping_template_test(string))
print(NIST.overlapping_template(string))
print(NIST.universal_statistical_test(string))
print(NIST.linear_complexity(string))
print(NIST.serial(string))
print(NIST.approximate_entropy(string))
print(NIST.cumulative_sums(string))
print(NIST.random_excursions_variant(string))
