import numpy as np
from my_stat import array_mean, standart_dev, data_in_sigma
np.set_printoptions(suppress=True) # don't use scientific notation

raw_data_10sec = 0
raw_data_20sec = np.genfromtxt('raw_data_20sec.csv', delimiter=',').reshape(200)

def unite_data(data, unite_n):
    if data.size % unite_n != 0: return "ERROR"
    data_united = np.zeros(data.size // unite_n)
    for i in range(data_united.size):
        for j in range(unite_n):
            data_united[i] += data[i * unite_n + j]
    return data_united

#в дате не должно быть нулей!!!
def table_for_hist(data):
    data_sorted = np.sort(data)
    data_size = data.size
    rows = int(max(data)) - int(min(data)) + 1
    table = np.zeros((rows, 3))
    for i in range(data_size):
        for j in range(rows):
            if data_sorted[i] == table[j][0]:
                table[j][1] += 1
                break
            elif table[j][0] == 0:
                table[j][0] = data_sorted[i]
                table[j][1] += 1
                break
    for i in range(rows):
        table[i][2] = table[i][1] / data_size
    return table

mean_20sec = array_mean(raw_data_20sec)
st_dev_20sec = standart_dev(raw_data_20sec)
sigma1_20sec = data_in_sigma(raw_data_20sec, st_dev_20sec, 1)
print("20sec среднее: " + str(mean_20sec) + "; стандартное отклонение: " + str(st_dev_20sec) + "; попало в 1сигму: " + str(sigma1_20sec))

raw_data_40sec = unite_data(raw_data_20sec, 2)
mean_40sec = array_mean(raw_data_40sec)
st_dev_40sec = standart_dev(raw_data_40sec)
sigma1_40sec = data_in_sigma(raw_data_40sec, st_dev_40sec, 1)
sigma2_40sec = data_in_sigma(raw_data_40sec, st_dev_40sec, 2)
print("40sec среднее: " + str(mean_40sec) + "; стандартное отклонение: " + str(st_dev_40sec) + "; попало в 1сигму: " + str(sigma1_40sec) + "; в 2сигмы: " + str(sigma2_40sec))

'''
#print(raw_data_40sec.reshape(10, 10))
table_40s = table_for_hist(raw_data_40sec)
print(table_40s)
'''