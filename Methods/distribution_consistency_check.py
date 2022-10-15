from scipy.stats import chi2
from numba import njit, prange


@njit(fastmath=True, nopython=True, parallel=True)
def tk(k, array, l):
    t_k = 0
    for i in prange(2 ** k):
        t_k += ((2 ** k * array[i] - l // k) ** 2) / ((2 ** k) * (l // k))
    return t_k

@njit(fastmath=True, nopython=True, parallel=True)
def solve_distribution_consistency_check(bin_data: list):
    k_list = []
    v_i_new = []
    t_k_new = []
    for k in prange(2, 17):

        v_i = []
        for i in range(2**k):
            summ = 0
            for j in prange(1, len(bin_data)//k):
                summ_int = 0
                for z in prange((j-1)*k, j*k):
                    summ_int += 2**(z-(j-1)*k) * bin_data[z]
                if summ_int == i:

                    summ += 1
            v_i.append(summ)

        t_k = tk(k, v_i, len(bin_data))

        k_list.append(k)
        v_i_new.append(v_i)
        t_k_new.append(t_k)
        print(k)
    print('End')
    return (k_list, v_i_new, t_k_new)


def distribution_consistency_check(bin_data: list, path: str, alpha=0.005):
    k_list, v_i_new, t_k_new = solve_distribution_consistency_check(bin_data)
    for i in range(len(k_list)):
        p_value = 1 - chi2.cdf(t_k_new[i], (2 ** k_list[i]) - 1)
        if alpha > p_value:
            open(path, 'a').write(
                f'------------ \nПроверка согласия распределения числа  \nUnsuccess p_value = {p_value}, k = {k_list[i]}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------ \nПроверка согласия распределения числа  \nSuccess p_value = {p_value}, k = {k_list[i]}\n------------\n')
