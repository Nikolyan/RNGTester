from numpy import sum, array, inf
from scipy.stats import chi2
from numba import njit, prange


def v_t_j(bin_data: list, t: int, k: int, j: int):
    return bin_data[(t) * k:(t) * k + k].count(j)


def solve_homogeneity_of_signs(bin_data: list):
    k = inf
    for s in range(1, len(bin_data) // 2 + 1):
        flag = True
        for j in range(1, len(bin_data) // s):
            z_j = array(bin_data[(j - 1) * s:j * s + 1])
            summ = sum(z_j)
            if min(s - summ, summ) < 100:
                flag = False
                break
        if flag:
            k = min(k, s)
            break
    u_0 = 0
    u_1 = 0
    v_i_0 = []
    v_i_1 = []
    for i in range(len(bin_data)//k+1):
        v_i_0.append(v_t_j(bin_data, i, k, 0))
        v_i_1.append(v_t_j(bin_data, i, k, 1))
        u_0 += v_i_0[i]
        u_1 += v_i_1[i]
    t_k = 0
    for i in range(len(bin_data)//k+1):
        t_k += ((v_i_0[i] - (u_0/(len(bin_data)//k)))**2)/u_0
        t_k += ((v_i_1[i] - (u_1/(len(bin_data)//k)))**2)/u_1
    t_k *= len(bin_data)//k

    return t_k, k


def homogeneity_of_signs(bin_data: list, path: str, alpha=0.005):
    t_k, k = solve_homogeneity_of_signs(bin_data)
    p_value = 1 - chi2.cdf(t_k, (len(bin_data) // k - 1))

    if alpha > p_value:
        open(path, 'a').write(
            f'------------ \nПроверка гипотезы однородности знаков  \nUnsuccess p_value = {p_value}, k = {k}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nПроверка гипотезы однородности знаков  \nSuccess p_value = {p_value}, k = {k}\n------------\n')

    return 0
