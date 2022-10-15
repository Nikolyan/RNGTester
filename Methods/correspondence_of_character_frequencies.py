import scipy
from math import sqrt


def correspondence_of_character_frequencies(bin_data: list, path: str, alpha=0.005):
    '''Проверка соответствия частот знаков в выходной последовательности теоретико-вероятностной модели образца
    ФГСЧ
    '''
    delta_min = 1 / 2 - (scipy.stats.norm.ppf([1 - (alpha / 2)], 0, 1)[0]) / (2 * sqrt(len(bin_data)))
    delta_max = 1 / 2 + (scipy.stats.norm.ppf([1 - (alpha / 2)], 0, 1)[0]) / (2 * sqrt(len(bin_data)))
    T_stat = bin_data.count(1) / len(bin_data)
    if delta_min <= T_stat <= delta_max:
        open(path, 'a').write(
            f'------------ \nПроверка соответствия частот знаков \nSuccess T-stat = {T_stat}\n------------\n')
    else:
        open(path, 'a').write(
            f'------------ \nПроверка соответствия частот знаков \nUnsuccess T-stat = {T_stat}\n------------\n')
    return 0
