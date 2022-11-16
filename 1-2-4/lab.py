import numpy as np
import matplotlib.pyplot as plt

def np_to_latex(data, centering="r"):
    print("\\begin{tabular}{|" + len(data[0]) * (centering + "|") + "}")
    print("\\hline")
    for row in data:
        print(*row, sep=" & ", end=" \\\\\n")
        print("\\hline")
    print("\\end{tabular}")

def find_side(side, name="a"):
    a = np.mean(side)
    s_a = np.std(side, ddof=1) / len(side) ** 0.5
    s_a = (s_a ** 2 + 0.05 ** 2) ** 0.5
    print("{} = {:.3f} \u00b1 {:.3f} мм".format(name, a, s_a))
    return a

cube = np.array([92.4, 92.6, 92.7, 92.6, 92.7, 92.8, 92.7, 92.8, 92.6, 92.7])
cube_a = find_side(cube, "cube")
parallelepiped_a = np.array([150.3, 150.3, 150.3, 150.3, 150.3, 150.4, 150.3, 150.3, 150.3, 150.3])
a = find_side(parallelepiped_a, "a")
parallelepiped_b = np.array([100.3, 100.4, 100.5, 100.3, 100.4, 100.4, 100.5, 100.5, 100.5, 100.4])
b = find_side(parallelepiped_b, "b")
parallelepiped_c = np.array([50.4, 50.5, 50.5, 50.6, 50.6, 50.5, 50.6, 50.5, 50.5, 50.5])
c = find_side(parallelepiped_c, "c")

#np_to_latex([range(1, 11), cube])
#np_to_latex([range(1, 11), parallelepiped_a, parallelepiped_b, parallelepiped_c])

periods = np.loadtxt('1-2-4/periods.csv', delimiter=',')
#print(periods)

periods = np.append(periods, np.round(np.mean(periods, 1, keepdims=True), 2), 1)
periods_mean = [row[-1] for row in periods]
#[2.58, 2.57, 3.05, 3.04, 3.02, 3.19, 3.71, 3.98, 3.29, 3.77, 3.37, 3.41]

def check_zero(lh_period, rh_periods, sides):
    rh_sum = np.sum(np.square(np.array(sides) * np.array(rh_periods)))
    difference = np.sum(np.square(np.array(sides))) * lh_period ** 2 - rh_sum
    sigma = 2 * 0.03 * ( (lh_period * np.sum(np.square(np.array(sides))))**2 + np.sum(np.square(np.square(np.array(sides)) * np.array(rh_periods))) ) ** 0.5
    return difference, sigma

'''
print(np.round(check_zero(periods_mean[-1], periods_mean[5:8], [a, b, c]), 0))
print(np.round(check_zero(periods_mean[-3], periods_mean[6:8], [b, c]), 0))
print(np.round(check_zero(periods_mean[8], periods_mean[5:8:2], [a, c]), 0))
print(np.round(check_zero(periods_mean[-2], periods_mean[5:7], [a, b]), 0))
'''

#first_table = periods[0:5].T
#second_table = periods[5:].T
#np_to_latex(first_table, "c")
#np_to_latex(second_table, "c")


##=== построение эллипсоида инерции ===##

frame_period = np.mean(periods_mean[0:2])
ellipsoid = []
for period in periods_mean[2:]:
    ellipsoid.append(1 / (period**2 - frame_period**2) ** 0.5)

#https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
def create_points(main_1, main_2, side_1, side_2, third):
    angle = np.arctan(side_2 / side_1)
    cos = np.cos(angle)
    sin = np.sin(angle)
    xs = np.reshape(np.array([main_1, -main_1, 0, 0, third * cos, third * cos, -third * cos, -third * cos]), (8, 1))
    ys = np.reshape(np.array([0, 0, main_2, -main_2, third * sin, -third * sin, third * sin, -third * sin]), (8, 1))
    return xs, ys


def create_ellipsis(ax, X, Y):
    ax.set_aspect('equal', adjustable='box')
    # Formulate and solve the least squares problem ||Ax - B||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    B = np.ones_like(X)
    x = np.linalg.lstsq(A, B)[0].squeeze()

    # Plot the points
    ax.scatter(X, Y, label='Экспериментальные данные')

    # Plot the least squares ellipse
    x_coord = np.linspace(-1, 1, 100)
    y_coord = np.linspace(-0.75, 0.75, 100)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
    ax.legend()
    ax.grid()

def create_circle(ax, X, Y, color, label):
    ax.set_aspect('equal', adjustable='box')
    # Formulate and solve the least squares problem ||Ax - B||^2
    A = np.hstack([X**2, Y**2])
    B = np.ones_like(X)
    x = np.linalg.lstsq(A, B)[0].squeeze()

    x_coord = np.linspace(-1, 1, 100)
    y_coord = np.linspace(-0.75, 0.75, 100)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * Y_coord**2
    ax.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=(color), linewidths=1.5)
    ax.scatter(X, Y, label=label)
    ax.legend()
    ax.grid()

fig, ax = plt.subplots()

def create_points_for_cube(main_1, main_2, side_1, side_2):
    angle = np.arctan(side_2 / side_1)
    cos = np.cos(angle)
    sin = np.sin(angle)
    xs = np.reshape(np.array([main_1, -main_1, main_2 * cos, -main_2 * cos]), (4, 1))
    ys = np.reshape(np.array([0, 0, main_2 * sin, -main_2 * sin]), (4, 1))
    return xs, ys

X, Y = create_points_for_cube(ellipsoid[0], ellipsoid[1], cube_a, cube_a * 3**0.5)
create_circle(ax, X, Y, 'r', "Плоскость AA'DD'")

X, Y = create_points_for_cube(ellipsoid[1], ellipsoid[2], cube_a * 3**0.5, cube_a * 2**0.5)
create_circle(ax, X, Y, 'y', "Плоскость EE'DD'")

X, Y = create_points_for_cube(ellipsoid[0], ellipsoid[2], cube_a, cube_a * 2**0.5)
create_circle(ax, X, Y, 'b', "Плоскость AA'EE'")

plt.savefig('1-2-4/cube', dpi=300, bbox_inches = 'tight')
plt.show()
'''
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Extract x coords and y coords of the ellipse as column vectors

def add_arrows(ax, el_1, el_2):
    ax.annotate(text='', xy=(-el_1, -0.5), xytext=(el_1, -0.5), arrowprops=dict(arrowstyle='<->'))
    ax.text(0, -0.5, "{:.2f}".format(2*el_1), ha='center', va='center', bbox={'boxstyle': 'square', 'facecolor': 'w', 'pad': 0.5})
    ax.annotate(text='', xy=(-0.75, -el_2), xytext=(-0.75, el_2), arrowprops=dict(arrowstyle='<->'))
    ax.text(-0.75, 0, "{:.2f}".format(2*el_2), ha='center', va='center', bbox={'boxstyle': 'square', 'facecolor': 'w', 'pad': 0.5})

X, Y = create_points(ellipsoid[3], ellipsoid[4], a, b, ellipsoid[8])
create_ellipsis(axes[0], X, Y)
axes[0].set_title("Главная плоскость xOy")
axes[0].text(0, 0, "$I_y:I_x = {:.2f}$".format((ellipsoid[3]/ellipsoid[4])**2), ha='center', va='center', bbox={'boxstyle': 'square', 'facecolor': 'b', 'alpha': 0.25, 'pad': 0.5})
add_arrows(axes[0], ellipsoid[3], ellipsoid[4])

X, Y = create_points(ellipsoid[4], ellipsoid[5], b, c, ellipsoid[7])
create_ellipsis(axes[1], X, Y)
axes[1].set_title("Главная плоскость yOz")
axes[1].text(0, 0, "$I_z:I_y = {:.2f}$".format((ellipsoid[4]/ellipsoid[5])**2), ha='center', va='center', bbox={'boxstyle': 'square', 'facecolor': 'b', 'alpha': 0.25, 'pad': 0.5})
add_arrows(axes[1], ellipsoid[4], ellipsoid[5])

X, Y = create_points(ellipsoid[3], ellipsoid[5], a, c, ellipsoid[6])
create_ellipsis(axes[2], X, Y)
axes[2].set_title("Главная плоскость xOz")
axes[2].text(0, 0, "$I_z:I_x = {:.2f}$".format((ellipsoid[3]/ellipsoid[5])**2), ha='center', va='center', bbox={'boxstyle': 'square', 'facecolor': 'b', 'alpha': 0.25, 'pad': 0.5})
add_arrows(axes[2], ellipsoid[3], ellipsoid[5])

plt.savefig('1-2-4/parped', dpi=300, bbox_inches = 'tight')
plt.show()
'''