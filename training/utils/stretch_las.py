import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np

def scale_to_nearest_grid(x, y, grid_size=250):
    """
    Scales x, y to the nearest multiple of grid_size (minimum = grid_size).
    Preserves the relative distribution of points.

    Returns:
        x_scaled, y_scaled, new_X_max, new_Y_max
    """
    X_max = np.max(x)
    Y_max = np.max(y)

    # Calculate the scaling factor, ensuring at least 1 * grid_size
    factor_X = max(np.round(X_max / grid_size), 1)
    factor_Y = max(np.round(Y_max / grid_size), 1)

    new_X_max = factor_X * grid_size-1 # -1 to be shure that it's not above k*250
    new_Y_max = factor_Y * grid_size-1

    # Scale points proportionally to fit into the new grid
    x_scaled = x * (new_X_max / X_max)
    y_scaled = y * (new_Y_max / Y_max)

    return x_scaled, y_scaled


def compute_r(theta, a, b, tol=1e-8):
    r_values = np.zeros_like(theta)
    for i, t in enumerate(theta):
        # Обработка особых значений
        if np.isclose(t % (2*np.pi), 0, atol=tol):
            r_values[i] = a
        elif np.isclose(t % (2*np.pi), np.pi/2, atol=tol):
            r_values[i] = b
        elif np.isclose(t % (2*np.pi), np.pi, atol=tol):
            r_values[i] = a
        elif np.isclose(t % (2*np.pi), 3*np.pi/2, atol=tol):
            r_values[i] = b
        else:
            r_values[i] = min(a / abs(np.cos(t)), b / abs(np.sin(t)))
    return r_values

def cartesian_to_polar(x, y):
    """
    Преобразует массивы координат (x, y) в полярные координаты (r, phi).
    phi возвращается в диапазоне [-pi, pi].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)      # [-π, π]
    phi = (phi + 2*np.pi) % (2*np.pi)   # [0, 2π]
    return r, phi

def compute_max_r_per_phi_bin(r, phi, phi_b):
    """
    Для каждого интервала углов [phi_b[k], phi_b[k+1]) находит максимум r.
    """
    R = []
    phi = phi % (2 * np.pi)  # гарантируем диапазон [0, 2pi)

    for k in range(len(phi_b) - 1):
        phi_start = phi_b[k]
        phi_end = phi_b[k+1]

        # Для учета перехода через 2pi
        if phi_end > phi_start:
            mask = (phi >= phi_start) & (phi < phi_end)
        else:
            # Например, [5.5, 0.2)
            mask = (phi >= phi_start) | (phi < phi_end)

        if np.any(mask):
            R_k = np.max(r[mask])
        else:
            R_k = 0.0  # или np.nan, если нужен маркер отсутствия точек
        R.append(R_k)

    return np.array(R)

def circular_linear_interp(phi_c, g):
    """
    Возвращает функцию для линейной интерполяции g по phi_c на окружности [0, 2pi).
    """
    # Сортируем phi_c и g по углу для корректной интерполяции
    sort_idx = np.argsort(phi_c)
    phi_c_sorted = phi_c[sort_idx]
    g_sorted = g[sort_idx]

    # Добавляем точку 2*pi для замыкания окружности
    phi_extended = np.append(phi_c_sorted, phi_c_sorted[0] + 2*np.pi)
    g_extended = np.append(g_sorted, g_sorted[0])

    # Создаем интерполяцию
    interp_func = interp1d(
        phi_extended,
        g_extended,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'  # безопасно, т.к. у нас замкнутая окружность
    )

    def wrapped_interp(phi):
        phi = np.mod(phi, 2*np.pi)
        return interp_func(phi)

    return wrapped_interp

def shift_to_center(x, y, is_plot=False):
        # 1. Построение охватывающего прямоугольника
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # 2. Вычисление центра и смещение всех точек так, чтобы центр стал (0,0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        x_shifted = x - x_center
        y_shifted = y - y_center

        # 3. Вычисление половины сторон прямоугольника
        a = (x_max - x_min) / 2
        b = (y_max - y_min) / 2

        print(f"Half-width a = {a:.4f}")
        print(f"Half-height b = {b:.4f}")

        if is_plot:
            # 4. Визуализация
            plt.figure(figsize=(6,6))
            plt.scatter(x_shifted, y_shifted, s=10, label='Shifted Points')

            # Рисуем прямоугольник
            rectangle = plt.Rectangle((-a, -b), 2*a, 2*b,
                                    edgecolor='red', facecolor='none', lw=2, label='Bounding Rectangle')
            plt.gca().add_patch(rectangle)

            plt.axis('equal')
            plt.grid(True)
            plt.title('Bounding Rectangle Centered at (0,0)')
            plt.legend()
            plt.show()

        return x_shifted, y_shifted, a, b

    
