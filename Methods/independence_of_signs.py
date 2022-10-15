from scipy.stats import chi2
from numba import njit, prange


@njit(fastmath=True, nopython=True, parallel=True)
def gen_i(n):
    res = []
    for i in prange(2 ** n):
        s = []
        for j in prange(n):
            s.append(i % 2)
            i = i // 2
        res.append(s)
    return res


@njit(fastmath=True, nopython=True, parallel=True)
def v_ind(bin_data: list, i_arr_elem: list, k: int, j: int):
    s = 0
    for t in prange(len(bin_data) - k):
        flag = True
        for p in prange(k):
            if bin_data[p + t] != i_arr_elem[p]:
                flag = False
                break
        if bin_data[t + k] != j:
            flag = False
        if flag:
            s += 1
    return s


@njit(fastmath=True, nopython=True, parallel=True)
def solve_independence_of_signs(bin_data: list):
    k_list = []
    i_arr_new = []
    u_0_new = []
    u_1_new = []
    v_i_0_new = []
    v_i_1_new = []
    v_i_new = []
    for k in prange(1, 25):
        u_0 = 0
        u_1 = 0
        i_arr = gen_i(k)
        v_i_0 = []
        v_i_1 = []
        v_i = []
        flag = True
        i_arr_new.append(len(i_arr))
        for i in prange(len(i_arr)):
            v_i_0.append(v_ind(bin_data, i_arr[i], k, 0))
            v_i_1.append(v_ind(bin_data, i_arr[i], k, 1))
            v_i.append(v_i_0[i] + v_i_1[i])
            u_0 += v_i_0[i]
            u_1 += v_i_1[i]
            if v_i_0[i] < 20 or v_i_1[i] < 20:
                flag = False
                break
        u_0_new.append(u_0)
        u_1_new.append(u_1)
        v_i_0_new.append(v_i_0)
        v_i_1_new.append(v_i_1)
        v_i_new.append(v_i)
        if not flag:
            print('Конец')
            break
        print(k)
        k_list.append(k)
    return (v_i_0_new, v_i_1_new, v_i_new, u_0_new, u_1_new, k_list, i_arr_new)


def independence_of_signs(bin_data: list, path: str, alpha=0.005):
    v_i_0, v_i_1, v_i, u_0, u_1, k_list, i_arr_new = solve_independence_of_signs(bin_data)

    for i in range(len(k_list)):
        t = 0
        for j in range(i_arr_new[i]):
            t += ((len(bin_data) * v_i_0[i][j] - v_i[i][j] * u_0[i]) ** 2) / (v_i[i][j] * u_0[i])
            t += ((len(bin_data) * v_i_1[i][j] - v_i[i][j] * u_1[i]) ** 2) / (v_i[i][j] * u_1[i])

        print('Считаю Чи')
        p_value = 1 - chi2.cdf(t / len(bin_data), (2 ** k_list[i]) - 1)
        if alpha > p_value:
            open(path, 'a').write(
                f'------------ \nПроверка гипотезы независимости знаков \nUnsuccess p_value = {p_value}, k = {k_list[i]}\n------------\n')
        else:
            open(path, 'a').write(
                f'------------ \nПроверка гипотезы независимости знаков \nSuccess p_value = {p_value}, k = {k_list[i]}\n------------\n')
    return 0