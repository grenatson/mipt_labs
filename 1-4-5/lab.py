import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_stat import LeastSquares

one_ten = np.arange(1, 11)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid(axis='y')

string_data = np.loadtxt('1-4-5/string_data.csv', delimiter=',')
tensions = np.zeros(np.shape(string_data)[0])
squared_speeds = np.zeros(np.shape(string_data)[0])

for i in range(np.shape(string_data)[0]):
    tension = string_data[i][0]
    freq = string_data[i][1:]
    
    ax1.scatter(one_ten, freq, s=8)
    mnk = LeastSquares(one_ten, freq, True)
    mnk.add_to_axes(ax1, start = 0, param_dict={'label': "T = {}".format(tension)})

    tensions[i] = tension
    squared_speeds[i] = mnk.k ** 2

    print("T = {}, linear approximation {}".format(tension, mnk))

ax2.scatter(tensions, squared_speeds)
mnk = LeastSquares(tensions, squared_speeds, True)
mnk.add_to_axes(ax2, start=0)
print("Density = {}".format(10 ** 6 / mnk.k))

ax1.legend()
ax2.grid()
plt.show()

'''
T1_data = string_data[0][1:]
T1 = string_data[0][0]
T2_data = string_data[1][1:]
T2 = string_data[1][0]
T3_data = string_data[2][1:]
T3 = string_data[2][0]
T4_data = string_data[3][1:]
T4 = string_data[3][0]

one_ten = np.arange(1, 11)

fig, ax = plt.subplots()
ax.grid(axis='y')

ax.scatter(one_ten, T1_data, s=8)
mnk_1 = LeastSquares(one_ten, T1_data, True)
mnk_1.add_to_axes(ax, start = 0)
print(mnk_1)

ax.scatter(one_ten, T2_data, s=8)
mnk_2 = LeastSquares(one_ten, T2_data, True)
mnk_2.add_to_axes(ax, start = 0)
print(mnk_2)

ax.scatter(one_ten, T4_data, s=8)
mnk_4 = LeastSquares(one_ten, T4_data, True)
mnk_4.add_to_axes(ax, start = 0)
print(mnk_4)

plt.show()
'''