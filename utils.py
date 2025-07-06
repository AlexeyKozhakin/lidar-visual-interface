import os
#import streamlit as st
from PIL import Image

def show_existing_files(save_dir, name='las files', ext='.las'):
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(ext)]
    if existing_files:
        st.info(f"Generated {name} from last session:")

        for file in existing_files:
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])  # Three columns: filename, delete button, view button

            with col1:
                st.text(file)  # Display filename

            with col2:
                if st.button("❌", key=file+name):  # Delete button with unique key
                    file_path = os.path.join(save_dir, file)
                    os.remove(file_path)  # Delete file
                    st.rerun()  # Restart script to update file list

            # Add view button for images
            if ext == '.png':
                if f"show_{file}" not in st.session_state:
                    st.session_state[f"show_{file}"] = False  # Initialize state

                with col3:
                    if st.button("🔍", key='view_'+file+name):
                        st.session_state[f"show_{file}"] = not st.session_state[f"show_{file}"]

                if st.session_state[f"show_{file}"]:
                    file_path = os.path.join(save_dir, file)
                    image = Image.open(file_path)
                    with st.expander(f"View {file}", expanded=True):
                        st.image(image, caption=file, use_container_width=True)
                        if st.button("Close", key="close_" + file+name):
                            st.session_state[f"show_{file}"] = False
                            st.rerun()  # Restart script to hide image