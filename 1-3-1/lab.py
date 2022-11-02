import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from my_stat import LeastSquares

g = 9.81
#=#=# first part #=#=#
'''
data = np.sort(np.loadtxt('1-3-1/scale.csv', delimiter=','), 1)

h = 1.334
r = 0.015
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
l_0 = 0.503

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
    s_a = np.std(rod_a, ddof=1) / len(rod_a) ** 0.5
    s_a = (s_a ** 2 + 0.1 ** 2) ** 0.5
    b = np.mean(rod_b)
    s_b = np.std(rod_b, ddof=1) / len(rod_b) ** 0.5
    s_b = (s_b ** 2 + 0.01 ** 2) ** 0.5
    if need_print:
        print("{} rod: a = {:.3f} \u00b1 {:.3f} мм, b = {:.3f} \u00b1 {:.3f} мм".format(material, a, s_a, b, s_b))
    return {'name': material, 'a': a / 1000, 's_a': s_a / 1000, 'b': b / 1000, 's_b': s_b / 1000}

#найдём среднее и переведём в метры
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

#np_to_latex(metal_deflect.round(2))
#np_to_latex(brass_deflect.round(2))
#np_to_latex(wood_deflect.round(2))

fig, axd = plt.subplot_mosaic([['metal', 'metal_flip'], ['brass', 'wood']],
                              figsize=(10, 6), layout="constrained", gridspec_kw={'wspace': 0.08, 'hspace': 0.05})

def process_deflect(ax, data, data_flipped, title, param_dict={'marker': '^', 'c': 'c', 's': 28}, flipped_param_dict={'marker': 'v', 'c': 'm', 's': 24}):
    data /= 1000
    data_flipped /= 1000
    ax.scatter(data[0], data[1], label="неперевёрнутая", **param_dict)
    ax.scatter(data_flipped[0], data_flipped[1], label="перевёрнутая", **flipped_param_dict)
    mnk = LeastSquares(np.concatenate((data[0], data_flipped[0])), np.concatenate((data[1], data_flipped[1])), True)
    mnk.add_to_axes(ax, param_dict={'lw': 1, 'ls': '--', 'c': 'r'})
    ax.text(0.70, 0.08, "Результат аппроксимации:\nk = {:.2e} \u00b1 {:.2e}".format(mnk.k, mnk.sigma_k),
            ha='center', transform=ax.transAxes,
            bbox={'boxstyle': 'square', 'facecolor': 'g', 'alpha': 0.4, 'pad': 0.6})
    
    ax.set_xlabel("m, кг")
    ax.yaxis.set_major_formatter(lambda y, pos: "{:.1f}".format(y * 1000))
    ax.set_ylabel(r"$y_{max}$, мм")
    
    ax.set_title(title)
    ax.grid()
    ax.legend(loc="upper left", title="Балка")
    print("{} {}, epsilon = {}".format(title, mnk, mnk.epsilon_k))
    return mnk.k, mnk.sigma_k

m_k, m_sk = process_deflect(axd['metal'], np.array([metal_deflect[0], metal_deflect[1]]), np.array([metal_deflect[2], metal_deflect[3]]), title="Металл, несмещённый")
mf_k, mf_sk = process_deflect(axd['metal_flip'], np.array([metal_deflect[4], metal_deflect[5]]), np.array([metal_deflect[6], metal_deflect[7]]), title="Металл, смещённый")

b_k, b_sk = process_deflect(axd['brass'], np.array([brass_deflect[0], brass_deflect[1]]), np.array([brass_deflect[2], brass_deflect[3]]), title="Латунь")
w_k, w_sk = process_deflect(axd['wood'], np.array([wood_deflect[0], wood_deflect[1]]), np.array([wood_deflect[2], wood_deflect[3]]), title="Дерево")


def get_s_k(k, s_k):
    return k * np.sqrt((s_k / k) ** 2 + (0.01 / 1) ** 2 + (1 / 1000) ** 2)

def get_E(k, s_k, rod):
    s_k = get_s_k(k, s_k)
    E = g / 4 * l_0 ** 3 / rod['a'] / k / rod['b'] ** 3
    sigma_E = E * np.sqrt((s_k / k) ** 2 + (rod['s_a'] / rod['a']) ** 2 + (3 * rod['s_b'] / rod['b']) ** 2 + (3 * 0.001 / l_0) ** 2) 
    return E, sigma_E, sigma_E / E

row = "{}: E = {:.6e} \u00b1 {:.6e} (epsilon = {:.6f})".format
print(row("metal", *get_E(m_k, m_sk, metal_rod)))
print(row("metal flipped", *get_E(mf_k, mf_sk, metal_rod)))
print(row("brass", *get_E(b_k, b_sk, brass_rod)))
print(row("wood", *get_E(w_k, w_sk, wooden_rod)))

plt.savefig('1-3-1/second_part.png', dpi=600)
plt.show()