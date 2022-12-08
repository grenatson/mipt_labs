import numpy as np
import matplotlib.pyplot as plt
from my_stat import LeastSquares

m1 = 93
t1 = np.array([106.14, 110.27, 110.24, 110.17, 110.13])
m2 = 142
t2 = np.array([71.77, 71.22, 71.52, 71.39, 71.46])
m3 = 173
t3 = np.array([59.02, 59.12, 59.55, 59.54, 59.22])
m4 = 215
t4 = np.array([47.18, 47.24, 47.40, 47.28, 47.34])
m5 = 268
t5 = np.array([37.69, 37.62, 37.48, 37.64, 37.67])
m6 = 338
t6 = np.array([30.09, 30.15, 30.02, 30.08, 30.07])

ms = np.array([m1, m2, m3, m4, m5, m6]) / 1000
ts = np.array([t1, t2, t3, t4, t5, t6])
sigmas_t = np.array([np.std(t, ddof=1) for t in ts])
ts = np.array([np.mean(t) for t in ts])
omegas = 2 * np.pi / ts
print(omegas)
l = 122 / 1000 #может что-то другое

torque = ms * 9.81 * l

fig, ax = plt.subplots()
ax.set_xlabel("$M, Н \cdot м$")
ax.set_ylabel("$\Omega, рад \cdot с^{-1}$")
ax.set_xlim((0, 0.5))
ax.set_ylim((0, 0.25))

ax.scatter(torque, omegas, label="Результаты измерений")

mnk = LeastSquares(torque, omegas, True)
mnk.add_to_axes(ax, 0, label="Прямая МНК")
print(mnk)

formula = "\Omega = \left(I_z \omega_0\\right)^{-1} \cdot M"
ax.text(0.65, 0.1, 
        "${}$ \n ${}$ = {:.4f} \u00b1 {:.4f}".format(formula, "\left(I_z \omega_0\\right)^{-1}", mnk.k, mnk.sigma_k),
        ha='center', transform=ax.transAxes, fontsize=12,
        bbox={'boxstyle': 'square', 'facecolor': 'r', 'alpha': 0.4, 'pad': 0.6})

plt.legend(loc='upper left')
plt.grid(True, linestyle="--")
plt.savefig("1-2-5/Omega-M.png", dpi=600)
plt.show()

#===I ротора===#

t_cylinder = np.array([4.05, 4.054, 4.053, 4.047])
t_rotor = np.array([3.244, 3.103, 3.225, 3.210, 3.230])
m_cylinder = 1.617
d_cylinder = 78 / 1000

sigma_cylinder = np.std(t_cylinder, ddof=1)
sigma_rotor = np.std(t_rotor, ddof=1)
t_cylinder = np.mean(t_cylinder)
t_rotor = np.mean(t_rotor)
I_cylinder = m_cylinder * d_cylinder**2 / 8
I_rotor = I_cylinder * t_rotor**2 / t_cylinder**2
print("cyliner: {} \u00b1 {}".format(t_cylinder, sigma_cylinder))
print("rotor: {} \u00b1 {}".format(t_rotor, sigma_rotor))
print("I_cylinder = {}, I_rotor = {}".format(I_cylinder, I_rotor))