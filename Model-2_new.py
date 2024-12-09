import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad


def vkb_transmission_probability(Ux, E, m, hbar):
    """
    Вычисляет вероятность прохождения через потенциальный барьер методом Вентцеля-Крамерса-Бриллюэна.

    Аргументы:
    Ux   : tuple (x, U) - кортеж, где x - координаты, U - значения потенциала на сетке
    E    : float - энергия частицы
    m    : float - масса частицы
    hbar : float - редуцированная постоянная Планка

    Возвращает:
    T    : float - вероятность прохождения барьера
    """
    x, U = Ux
    dx = x[1] - x[0]

    # Выделяем регион, где U(x) > E (барьерный регион)
    barrier_region = U > E

    # Если весь барьерный регион пустой (U(x) < E),
    if np.all(~barrier_region):
        print("Потенциал меньше энергии частицы по всему барьеру. Вероятность = 1.")
        return 1.0

    # Определение подынтегральной функции
    def integrand(x_value):
        index = np.searchsorted(x, x_value)
        if index == len(x):
            index -= 1
        U_value = U[index]
        return np.sqrt(2 * m * (U_value - E)) / hbar

    # Разделяем барьерный регион на два подинтервала, чтобы избежать проблем с интеграцией
    region_min = x[barrier_region].min()
    region_max = x[barrier_region].max()

    # Разделение на два подинтервала
    middle = (region_max + region_min) / 2
    integral_value_1, _ = quad(integrand, region_min, middle, limit=1000)
    integral_value_2, _ = quad(integrand, middle, region_max, limit=1000)

    # Суммируем результаты для двух частей
    integral_value = integral_value_1 + integral_value_2

    # Вероятность туннелирования (по методу Вентцеля-Крамерса-Бриллюэна)
    T = np.exp(-2 * integral_value)
    return T


# Пример использования
if __name__ == "__main__":
    hbar = 1.0545718e-34  # Планковская постоянная (Дж·с)
    m = 9.11e-31  # Масса частицы (кг)
    E = 1.9e-20  # Энергия частицы (Дж)

    # # Потенциальный барьер: U(x) = U0 * exp(-x^2 / sigma^2)
    # x = np.linspace(-5e-10, 5e-10, 1000)
    # U0 = 2e-20  # Высота барьера (Дж)
    # sigma = 1e-10  # Ширина барьера (м)
    # U = U0 * np.exp(-x ** 2 / sigma ** 2)

    # Прямоугольный потенциальный барьер
    x = np.linspace(-5e-10, 5e-10, 1000)
    U0 = 2e-20  # Высота барьера (Дж)
    x_min = -1e-10  # Начало барьера (м)
    x_max = 1e-10  # Конец барьера (м)
    U = np.where((x >= x_min) & (x <= x_max), U0, 0)

    T = vkb_transmission_probability((x, U), E, m, hbar)
    print(f"Вероятность прохождения методом ВКБ: {T:.6f}")

    # Построение графика U(x)
    plt.plot(x * 1e9, U, label="U(x)", color="blue")
    plt.axhline(E, color="red", linestyle="--", label=f"E = {E:.2e} Дж")
    plt.title("Потенциальный барьер и энергия частицы")
    plt.xlabel("x (нм)")
    plt.ylabel("U(x) (Дж)")
    plt.legend()
    plt.grid(True)
    plt.show()







