import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from my_stat import LeastSquares

#=#=# first part #=#=#

data = np.sort(np.loadtxt('1-3-1/scale.csv', delimiter=','), 1)

h = 1.334
r = 0.015
g = 9.81
l = 1.768
d = 0.46 / 1000
loads = data[0] / 1000
ns = data[1] / 100
print(loads, ns)

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(loads[3:], ns[3:], c='b', label='Основные данные')
ax.scatter(loads[:3], ns[:3], c='#aaaaff', marker='s', s=20, label='Нерассматриваемые данные')

#ax.xaxis.set_major_formatter(lambda x, pos: "{:.0f}".format(x * 1000))
ax.set_xlabel("m, кг")
ax.yaxis.set_major_formatter(lambda y, pos: "{:.0f}".format(y * 100))
ax.set_ylabel("n, см")

mnk = LeastSquares(loads[3:], ns[3:])
mnk.add_to_axes(ax, start=0, param_dict={'lw': 1.5, 'c': 'g', 'ls': '--', 'label': r'$n = \dfrac{1}{k} \cdot \dfrac{2hg}{r} m + const$'})

k = 2 * h * g / r / mnk.k
k_sigma = mnk.epsilon_k * k
ax.text(2.25, 0.15, "Результат аппроксимации:\nk = {:.2f} \u00b1 {:.2f} Н/м".format(k, k_sigma), ha='center', bbox={'boxstyle': 'square', 'facecolor': 'g', 'alpha': 0.5, 'pad': 0.75})
E = k * l / (np.pi * d ** 2 / 4)

print("Упругость: {} \u00b1 {}; модуль Юнга: {:.3e}".format(k, k_sigma, E))

ax.legend()
ax.grid()
plt.savefig('1-3-1/first_part.png', dpi=300)
plt.show()

'''
#=#=# second part #=#=#

def np_to_latex(data, centering="r"):
    print("\\begin{tabular}{|" + len(data[0]) * (centering + "|") + "}")
    print("\\hline")
    for row in data:
        print(*row, sep=" & ", end=" \\\\\n")
        print("\\hline")
    print("\\end{tabular}")

# rod dim
rods = np.loadtxt('1-3-1/rod.csv', delimiter=',')
#приведение к миллиметрам
rods[3] = np.round((rods[3] + 35) / 10, 2)
rods[1] = np.round((rods[1] + 35) / 10, 2)
rods[5] = np.round((rods[5] + 100) / 10, 2)
#np_to_latex(rods)

def get_rods(rod_a, rod_b, material="unknown", need_print=True):
    a = np.mean(rod_a)
    a_s = np.std(rod_a, ddof=1) / len(rod_a) ** 0.5
    b = np.mean(rod_b)
    b_s = np.std(rod_b, ddof=1) / len(rod_b) ** 0.5
    if need_print:
        print("{} rod: a = {:.3f} \u00b1 {:.3f}, b = {:.3f} \u00b1 {:.3f}".format(material, a, a_s, b, b_s))
    return {'name': material, 'a': a, 'a_s': a_s, 'b': b, 'b_s': b_s}

metal_rod = get_rods(rods[0], rods[1], material="metal")
brass_rod = get_rods(rods[2], rods[3], material="brass")
wooden_rod = get_rods(rods[4], rods[5], material="wooden")
#с балками покончили

metal_deflect = np.loadtxt('1-3-1/deflections.csv', delimiter=',', max_rows=8)
brass_deflect = np.loadtxt('1-3-1/deflections.csv', delimiter=',', skiprows=16, max_rows=4)
wood_deflect = np.loadtxt('1-3-1/deflections.csv', delimiter=',', skiprows=24, max_rows=4)

#исправления
m_0 = 54.5

metal_deflect[0::2] += m_0
brass_deflect[0::2] += m_0
wood_deflect[0::2] += m_0

metal_deflect[1] -= 0.41
metal_deflect[3] -= 0.08
metal_deflect[5] -= 0.44
metal_deflect[7] -= 0.06

brass_deflect[1] -= 0.5
brass_deflect[3] -= 0.05

wood_deflect[1] -= 0.94
wood_deflect[3] -= 0.61

metal_deflect.round(2)

np_to_latex(metal_deflect.round(2))
np_to_latex(brass_deflect.round(2))
np_to_latex(wood_deflect.round(2))
'''