import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

freq = np.array([400, 398, 396, 395, 393, 390, 401, 402])
ampl = np.array([52, 40, 26, 24, 12, 8, 24, 4])

fig, ax = plt.subplots()
ax.scatter(freq, ampl)
ax.hlines(0.7 * np.max(ampl), np.min(freq), np.max(freq), linestyles='--')

ax.grid()
plt.show()