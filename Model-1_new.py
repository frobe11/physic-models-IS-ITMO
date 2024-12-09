import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Константы
hbar = 1.0545718e-34  # Дж*с (постоянная Планка)
m = 9.10938356e-31    # кг (масса электрона)
eV = 1.60218e-19      # Дж (1 эВ в Джоулях)

# Параметры ямы
a = 1e-9  # Половина ширины ямы (1 нм)
U = 3 * eV  # Глубина ямы (5 эВ)
N = 1000    # Число точек для разбиения [-3a, 3a]
L = 3 * a   # Общая длина области

# Дискретизация пространства
x = np.linspace(-L, L, N)
dx = x[1] - x[0]  # Шаг разбиения

# Потенциал V(x)
V = np.zeros_like(x)
V[np.abs(x) < a] = -U

# Матрица Гамильтониана
main_diag = (hbar**2 / (m * dx**2)) + V
off_diag = -hbar**2 / (2 * m * dx**2) * np.ones(N-1)

# Нахождение собственных значений и функций
E, psi = eigh_tridiagonal(main_diag, off_diag)

# Индексы связанных состояний
bound_indices = np.where(E < 0)[0]


# Нормализация только для связанных состояний
normalized_wavefunctions = psi[:, bound_indices]/ np.sqrt(
    np.trapezoid(psi[:, bound_indices]**2, x*1e9, axis=0)
)


# Выводим связанные состояния
plt.figure(figsize=(12, 8))

for n, idx in enumerate(bound_indices):
    plt.plot(x * 1e9, normalized_wavefunctions[:, n] + E[idx] / eV,
             label=f"E[{idx+1}]={E[idx]/eV:.2f} эВ")

# Отображение потенциала
plt.plot(x * 1e9, V / eV, 'k--', label="Потенциал V(x)")

# Настройки графика
plt.title("Собственные функции и уровни энергии в прямоугольной потенциальной яме")
plt.xlabel("x, нм")
plt.ylabel("Энергия, Э, эВ")
# plt.legend()
plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left')
plt.grid()
plt.show()