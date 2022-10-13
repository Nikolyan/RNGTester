from creat_array import creat_array
from scipy.stats import chi2
from math import sqrt
from tqdm import tqdm, trange

path = r"C:\Users\Илья\Desktop\не трогать\dt.bin"
alpha_const = 0.005
ar = creat_array(path, 100000)


def func1(bin_data: list, alpha=alpha_const):
    '''Проверка соответствия частот знаков в выходной последовательности теоретико-вероятностной модели образца
    ФГСЧ
    '''
    delta_min = 1 / 2 - (scipy.stats.norm.ppf([1 - (alpha / 2)], 0, 1)[0]) / (2 * sqrt(len(bin_data)))
    delta_max = 1 / 2 + (scipy.stats.norm.ppf([1 - (alpha / 2)], 0, 1)[0]) / (2 * sqrt(len(bin_data)))
    T_stat = bin_data.count(1) / len(bin_data)
    if delta_min <= T_stat <= delta_max:
        return f'------------ \nПроверка соответствия частот знаков \nSuccess T-stat = {T_stat}\n------------'
    else:
        return f'------------ \nПроверка соответствия частот знаков \nUnsuccess T-stat = {T_stat}\n------------'


def func2(k_max: int, bin_data: list, alpha=alpha_const):
    for k in range(1,k_max+1):
        u_0 = 0
        u_1 = 0
        i_arr = gen_i(k)
        v_i_0 = []
        v_i_1 = []
        v_i = []
        for i in trange(len(i_arr)):
            v_i_0.append(v_ind(bin_data, i_arr[i], k, 0))
            v_i_1.append(v_ind(bin_data, i_arr[i], k, 1))
            v_i.append(v_i_0[i] + v_i_1[i])
            u_0 += v_i_0[i]
            u_1 += v_i_1[i]
        t = 0
        for i in range(len(i_arr)):
            t += ((len(bin_data) * v_i_0[i] - v_i[i]*u_0)**2)/(v_i[i]*u_0)
            t += ((len(bin_data) * v_i_1[i] - v_i[i] * u_1) ** 2) / (v_i[i] * u_1)

        t = t/len(bin_data)
        p_value = 1 - chi2.cdf(t, (2**k)-1)
        if alpha > p_value:
            print(f'------------ \nПроверка гипотезы независимости знаков \nUnsuccess p_value = {p_value}, k = {k}\n------------')
        else:
            print(
                f'------------ \nПроверка гипотезы независимости знаков \nSuccess p_value = {p_value}, k = {k}\n------------')

def v_ind(bin_data: list, i_arr_elem: list, k: int, j: int):
    s = 0
    for t in range(len(bin_data) - k):
        flag = True
        for p in range(k):
            if bin_data[p + t] != i_arr_elem[p]:
                flag = False
                break
        if bin_data[t + k] != j:
            flag = False
        if flag:
            s += 1
    return s

def gen_i(n):
    res = []
    for i in range(2 ** n):
        s = []
        for j in range(n):
            s.append(i % 2)
            i = i // 2
        res.append(s)
    return res




