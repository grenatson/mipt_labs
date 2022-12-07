import numpy as np
import matplotlib.pyplot as plt
from my_stat import LeastSquares, np_to_latex

a = np.array([46, 44, 40, 38, 34, 30, 25, 18, 12, 6]) #сантиметры
a = a / 100
t = np.array([32.2, 31.9, 31.32, 31.16, 30.81, 30.50, 30.55, 31.90, 35.45, 45.27])
T = t / 20 #периоды колебаний

'''
np_to_latex([a, t, T])

fig, ax = plt.subplots()
ax.scatter(a, T, label="Результаты эксперимента", s=16, c="r")
ax.plot(a, T, label="Экспериментальная зависимость $T(a)$", linestyle="--", color="green")
ax.set_xlabel("$a, м$")
ax.set_ylabel("$T, c$")
ax.set_xlim((0, 0.5))

x_start, x_end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(x_start, x_end, 0.05))

least_i = np.argpartition(T, 2)[:2]
ax.hlines(T[least_i[0]], x_start, x_end, colors="orangered", linestyles="--", linewidth=1)
ax.text(0.05, T[least_i[0]], "$T_{} = {:.2f}$ с".format("{min}", T[least_i[0]]), va="bottom")
ax.axvspan(a[least_i[0]], a[least_i[1]], color="red", alpha=0.25, label="Область минимума периода")

ax.grid(True, linestyle="--")
ax.legend()
plt.savefig("1-4-1/direct.png", dpi=600)
plt.show()
'''
#===# теперь линеаризованный график

T2a = T ** 2 * a
a2 = a ** 2

fig, ax = plt.subplots()
ax.scatter(a2, T2a, label="Результаты эксперимента", s=16, c="r")
ax.set_xlabel("$a^2, м^2$")
ax.set_ylabel("$T^2a, c^2 \cdot м$")

mnk = LeastSquares(a2[:-1], T2a[:-1])
print(mnk)
mnk.add_to_axes(ax, label="Линейная зависимость МНК")
formula = "T^2a = \\dfrac{4\\pi^2}{g}\\left( a^2 + \\dfrac{l^2}{12} \\right)"
ax.text(0.70, 0.1, 
        "${}$:\n ${}$ = {:.2f} \u00b1 {:.2f}\n ${}$ = {:.3f} \u00b1 {:.3f}".format(formula, "\\dfrac{4\pi^2}{g}", mnk.k, mnk.sigma_k, "\\dfrac{4\pi^2}{g} \\dfrac{l^2}{12}", mnk.b, mnk.sigma_b),
        ha='center', transform=ax.transAxes,
        bbox={'boxstyle': 'square', 'facecolor': 'g', 'alpha': 0.4, 'pad': 0.6})

ax.grid(True, linestyle="--")
ax.legend()
plt.savefig("1-4-1/linear.png", dpi=600)
plt.show()