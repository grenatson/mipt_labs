import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_stat import LeastSquares

T1_data = np.loadtxt('1-4-5\string_data.csv', delimiter=',', usecols=range(1, 11), max_rows=1, skiprows=0)
T2_data = np.loadtxt('1-4-5\string_data.csv', delimiter=',', usecols=range(1, 11), max_rows=1, skiprows=1)
T3_data = np.loadtxt('1-4-5\string_data.csv', delimiter=',', usecols=range(1, 11), max_rows=1, skiprows=2)
T4_data = np.loadtxt('1-4-5\string_data.csv', delimiter=',', usecols=range(1, 11), max_rows=1, skiprows=3)

one_ten = np.arange(1, 11)

fig, ax = plt.subplots()
ax.grid()

ax.scatter(one_ten, T1_data)
mnk_1 = LeastSquares(one_ten, T1_data, True)
mnk_1.add_to_axes(ax, start = 0)
print(mnk_1)

ax.scatter(one_ten, T2_data)
mnk_2 = LeastSquares(one_ten, T2_data, True)
mnk_2.add_to_axes(ax, start = 0)
print(mnk_2)

ax.scatter(one_ten, T4_data)
mnk_4 = LeastSquares(one_ten, T4_data, True)
mnk_4.add_to_axes(ax, start = 0)
print(mnk_4)

plt.show()