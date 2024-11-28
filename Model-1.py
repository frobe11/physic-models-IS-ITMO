import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import solve_bvp

# Константы
hbar = 1.0545718e-34  # Планковская постоянная (Дж·с)
m = 9.11e-31          # Масса частицы (например, электрона) (кг)
U = 15.0               # Глубина ямы (эВ)
a = 1e-9              # Ширина ямы (м)

# Перевод в подходящие единицы
U *= 1.60218e-19      # Джоуль
L = a             # Длина ямы

# Сетка координат
N = 500
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Потенциальная яма
V = np.zeros_like(x)
V[np.abs(x) > a] = 0
V[np.abs(x) <= a] = -U

# Гамильтониан
T = -0.5 * hbar**2 / m / dx**2
diagonal = -2 * T + V
off_diagonal = np.ones(N - 1) * T
H = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

# Собственные значения и функции
eigenvalues, eigenvectors = eigh(H)

# Связанные состояния
bound_states = eigenvalues[eigenvalues < 0]
formatted_energies = np.array2string(bound_states / 1.60218e-19, formatter={'float_kind': lambda x: f"{x:.4f}"})
print(f"Энергии связанных состояний (эВ): {formatted_energies}")

colors = ['r', 'g', 'b']
for i, eigenvalue in enumerate(bound_states[:3]):  # 3 первых состояния
    plt.plot(x, eigenvectors[:, i], label=f"n={i+1}, E={eigenvalue/1.60218e-19:.2f} эВ",
             color=colors[i], linewidth=2)
plt.title("Собственные функции прямоугольной ямы")
plt.xlabel("x (м)")
plt.ylabel("$\psi(x)$")
plt.legend()
plt.grid()
plt.show()


def transmission_probability(Ux, E, m, hbar):

    x, U = Ux
    dx = x[1] - x[0]  # Шаг сетки

    # Волновое число (k или \kappa)
    k_squared = 2 * m * (E - U) / hbar**2
    k_values = np.zeros_like(k_squared, dtype=complex)  # Инициализация волнового числа

    # Условная обработка положительных и отрицательных значений
    k_values[k_squared >= 0] = np.sqrt(k_squared[k_squared >= 0])  # Для \( E > U \)
    k_values[k_squared < 0] = 1j * np.sqrt(-k_squared[k_squared < 0])  # Для \( E < U \)

    # Инициализация матрицы передачи
    T_matrix = np.eye(2, dtype=complex)

    for i in range(1, len(x)):
        k_left = k_values[i - 1]
        k_right = k_values[i]

        # Избежание деления на ноль
        k_left = k_left if np.abs(k_left) > 1e-12 else 1e-12
        k_right = k_right if np.abs(k_right) > 1e-12 else 1e-12

        # Матрица перехода на границе
        M_interface = np.array([[0.5 * (1 + k_right / k_left), 0.5 * (1 - k_right / k_left)],
                                 [0.5 * (1 - k_right / k_left), 0.5 * (1 + k_right / k_left)]])

        # Матрица распространения
        M_propagation = np.array([[np.exp(-1j * k_right * dx), 0],
                                  [0, np.exp(1j * k_right * dx)]])

        # Обновление матрицы передачи
        T_matrix = M_propagation @ M_interface @ T_matrix

    # Вероятность прохождения
    T = np.abs(1 / T_matrix[0, 0])**2
    return T

if __name__ == "__main__":
    hbar = 1.0545718e-34
    m = 9.11e-31
    E = 1.5e-20

    # Потенциальный барьер: U(x) = U0 * exp(-x^2 / sigma^2)
    x = np.linspace(-5e-10, 5e-10, 1000)  # Сетка координат
    U0 = 2e-20                            # Высота барьера (Дж)
    sigma = 1e-10                         # Ширина барьера (м)
    U = U0 * np.exp(-x**2 / sigma**2)     # Потенциал

    T = transmission_probability((x, U), E, m, hbar)
    print(f"Вероятность прохождения: {T:.6f}")
