import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_stat import LeastSquares

one_ten = np.arange(1, 11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.grid(axis='y')

string_data = np.loadtxt('1-4-5/string_data.csv', delimiter=',')
tensions = np.zeros(np.shape(string_data)[0])
squared_speeds = np.zeros(np.shape(string_data)[0])

for i in range(np.shape(string_data)[0]):
    tension = string_data[i][0]
    freq = string_data[i][1:]
    
    ax1.scatter(one_ten / 0.97, freq, s=8)
    mnk = LeastSquares(one_ten / 0.97, freq, True)
    mnk.add_to_axes(ax1, start = 0, param_dict={'label': "T = {}".format(tension)})

    tensions[i] = tension
    #mnk.k *= 0.97
    squared_speeds[i] = mnk.k ** 2

    print("T = {}, linear approximation {}".format(tension, mnk))

ax2.scatter(tensions, squared_speeds)
mnk = LeastSquares(tensions, squared_speeds, True)
mnk.add_to_axes(ax2, start=0)
print(squared_speeds)
print("Density = {:.3f}".format(10 ** 6 / mnk.k))
print(mnk)

ax1.legend()
ax2.grid()
fig.savefig('1-4-5/graphs.png')
plt.show()