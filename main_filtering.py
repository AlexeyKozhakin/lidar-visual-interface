import streamlit as st
import laspy
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

# --- Функции для обработки ---
def get_las_info(las):
    points = len(las.points)
    dx = np.max(las.x) - np.min(las.x)
    dy = np.max(las.y) - np.min(las.y)
    dz = np.max(las.z) - np.min(las.z)
    return points, dx, dy, dz

def simple_cleaning(las, iterations=1):
    # Простейший пример очистки — удаление точек выше и ниже порога
    z_mean = np.mean(las.z)
    z_std = np.std(las.z)
    mask = (las.z > z_mean - 3*z_std) & (las.z < z_mean + 3*z_std)
    cleaned_las = las.points[mask]
    removed = len(las.points) - np.sum(mask)
    return cleaned_las, removed, iterations

# --- Интерфейс ---
st.title("Очистка и фильтрация LAS-файлов")

st.sidebar.header("Настройки фильтрации")

uploaded_file = st.file_uploader("Загрузите LAS-файл", type=["las"])

if uploaded_file:
    las = laspy.read(uploaded_file)
    st.subheader("Информация о загруженном файле")

    points, x_range, y_range, z_range = get_las_info(las)

    st.write(f"**Количество точек:** {points:,}".replace(",", " "))
    st.write(f"**X:** {x_range:.2f}")
    st.write(f"**Y:** {y_range:.2f}")
    st.write(f"**Z:** {z_range:.2f}")

    # --- Визуализация исходных данных ---
    if st.checkbox("Показать точки до фильтрации (XY)"):
        fig, ax = plt.subplots()
        ax.scatter(las.x[::100], las.y[::100], s=1, color='blue')
        ax.set_title("До фильтрации")
        st.pyplot(fig)

    # --- Настройки фильтрации ---
    st.sidebar.subheader("Выберите фильтры")
    apply_outlier = st.sidebar.checkbox("Удаление выбросов", value=True)
    apply_height = st.sidebar.checkbox("Фильтр по высоте")
    apply_density = st.sidebar.checkbox("Фильтр по плотности")

    iterations = st.sidebar.slider("Количество итераций очистки", 1, 5, 1)

    # --- Кнопка очистки ---
    if st.button("Очистить данные"):
        cleaned_points, removed, iters = simple_cleaning(las, iterations)

        st.subheader("Информация после очистки")
        st.write(f"**Количество точек:** {len(cleaned_points)}")
        st.write(f"**Удалено точек:** {removed}")
        st.write(f"**Количество итераций:** {iters}")

        x_clean = cleaned_points['X']
        y_clean = cleaned_points['Y']
        z_clean = cleaned_points['Z']
        st.write(f"**X:** ({x_clean.min()}, {x_clean.max()})")
        st.write(f"**Y:** ({y_clean.min()}, {y_clean.max()})")
        st.write(f"**Z:** ({z_clean.min()}, {z_clean.max()})")

        # --- Визуализация после очистки ---
        if st.checkbox("Показать точки после фильтрации (XY)"):
            fig, ax = plt.subplots()
            ax.scatter(x_clean[::100], y_clean[::100], s=1, color='green')
            ax.set_title("После фильтрации")
            st.pyplot(fig)

        # --- Сохранение результата ---
        if st.button("Сохранить очищенный файл"):
            output = io.BytesIO()
            header = las.header
            new_las = laspy.LasData(header)
            new_las.points = cleaned_points
            new_las.write(output)
            st.download_button(
                label="Скачать LAS",
                data=output.getvalue(),
                file_name="cleaned.las",
                mime="application/octet-stream"
            )
