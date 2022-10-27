import time

import scipy

from creat_array import creat_array
from scipy.stats import chi2
from math import sqrt
from tqdm import tqdm, trange
import numpy as np

path = r'C:\Users\nikol\Desktop\Pet-projects\dt.bin'
alpha_const = 0.005
ar = creat_array(path, 10000000)


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


def func2(bin_data=ar, alpha=alpha_const):

    for k in range(1, 25):
        u_0 = 0
        u_1 = 0
        i_arr = gen_i(k)
        v_i_0 = []
        v_i_1 = []
        v_i = []
        flag = True
        for i in trange(len(i_arr)):
            v_i_0.append(func2_v_ind(bin_data, i_arr[i], k, 0))
            v_i_1.append(func2_v_ind(bin_data, i_arr[i], k, 1))
            v_i.append(v_i_0[i] + v_i_1[i])
            u_0 += v_i_0[i]
            u_1 += v_i_1[i]
            if v_i_0[i] < 20 or v_i_1[i] < 20:
                flag = False
                break
        if not flag:
            print('Конец')
            break

        t = 0
        for i in range(len(i_arr)):
            t += ((len(bin_data) * v_i_0[i] - v_i[i] * u_0) ** 2) / (v_i[i] * u_0)
            t += ((len(bin_data) * v_i_1[i] - v_i[i] * u_1) ** 2) / (v_i[i] * u_1)

        t = t / len(bin_data)
        p_value = 1 - chi2.cdf(t, (2 ** k) - 1)
        if alpha > p_value:
            print(
                f'------------ \nПроверка гипотезы независимости знаков \nUnsuccess p_value = {p_value}, k = {k}\n------------')
        else:
            print(
                f'------------ \nПроверка гипотезы независимости знаков \nSuccess p_value = {p_value}, k = {k}\n------------')


def func2_v_ind(bin_data: list, i_arr_elem: list, k: int, j: int):
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


def func3(bin_data: list, alpha=alpha_const):
    k = 100000000000
    for s in range(1, len(bin_data) // 2 + 1):
        flag = True
        for j in range(1, len(bin_data) // s):
            z_j = np.array(bin_data[(j - 1) * s:j * s + 1])
            summ = np.sum(z_j)
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
        v_i_0.append(func3_v_t_j(bin_data, i, k, 0))
        v_i_1.append(func3_v_t_j(bin_data, i, k, 1))
        u_0 += v_i_0[i]
        u_1 += v_i_1[i]
    t_k = 0
    for i in range(len(bin_data)//k+1):
        t_k += ((v_i_0[i] - (u_0/(len(bin_data)//k)))**2)/u_0
        t_k += ((v_i_1[i] - (u_1/(len(bin_data)//k)))**2)/u_1
    t_k *= len(bin_data)//k
    p_value = 1 - chi2.cdf(t_k, (len(bin_data)//k - 1))
    if alpha > p_value:
        print(
            f'------------ \nПроверка гипотезы однородности знаков  \nUnsuccess p_value = {p_value}, k = {k}\n------------')
    else:
        print(
            f'------------ \nПроверка гипотезы однородности знаков  \nSuccess p_value = {p_value}, k = {k}\n------------')

def func3_v_t_j(bin_data: list, t: int, k: int, j: int):
    return bin_data[(t) * k:(t) * k + k].count(j)


def func4(bin_data: list, alpha=alpha_const):
    for k in range(2, 17):
        v_i = []
        for i in range(2**k):
            summ = 0
            for j in range(1, len(bin_data)//k):
                summ_int = 0
                for z in range((j-1)*k, j*k):
                    summ_int += 2**(z-(j-1)*k) * bin_data[z]
                if summ_int == i:

                    summ += 1
            v_i.append(summ)

        t_k = 0
        for i in range(2**k):
            t_k += ((2**k * v_i[i] - len(bin_data)//k)**2)/((2**k)*(len(bin_data)//k))
        p_value = 1 - chi2.cdf(t_k, (2**k)-1)
        if alpha > p_value:
            print(
                f'------------ \nПроверка согласия распределения числа  \nUnsuccess p_value = {p_value}, k = {k}\n------------')
        else:
            print(
                f'------------ \nПроверка согласия распределения числа  \nSuccess p_value = {p_value}, k = {k}\n------------')
