import os
import streamlit as st
from PIL import Image

def show_existing_files(save_dir, name='las files', ext='.las'):
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(ext)]
    if existing_files:
        st.info(f"Generated {name} from last session:")

        for file in existing_files:
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])  # Три колонки: имя файла, кнопка удаления, кнопка просмотра

            with col1:
                st.text(file)  # Отображаем имя файла

            with col2:
                if st.button("❌", key=file+name):  # Кнопка удаления с уникальным ключом
                    file_path = os.path.join(save_dir, file)
                    os.remove(file_path)  # Удаляем файл
                    st.rerun()  # Перезапускаем скрипт, чтобы обновить список файлов

            # Добавляем кнопку просмотра для изображений
            if ext == '.png':
                if f"show_{file}" not in st.session_state:
                    st.session_state[f"show_{file}"] = False  # Инициализация состояния

                with col3:
                    if st.button("🔍", key='view_'+file+name):
                        st.session_state[f"show_{file}"] = not st.session_state[f"show_{file}"]

                if st.session_state[f"show_{file}"]:
                    file_path = os.path.join(save_dir, file)
                    image = Image.open(file_path)
                    with st.expander(f"Просмотр {file}", expanded=True):
                        st.image(image, caption=file, use_container_width=True)
                        if st.button("Закрыть", key="close_" + file+name):
                            st.session_state[f"show_{file}"] = False
                            st.rerun()  # Перезапускаем скрипт для скрытия изображения