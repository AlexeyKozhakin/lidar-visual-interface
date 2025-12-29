# Симметричное дополнение

import os
import cv2
import numpy as np

def process_image(image_path):
    """
    Обрабатывает изображение по заданному алгоритму.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Ошибка загрузки: {image_path}")
        return None
    
    # Отражение и дополнение справа
    mirrored_h = cv2.flip(image, 1)
    row = np.hstack((image, mirrored_h))
    row = np.hstack((row, row))  # Дополнение справа еще раз
    
    # Отражение и дополнение снизу
    mirrored_v = cv2.flip(row, 0)
    final_image = np.vstack((row, mirrored_v))
    final_image = np.vstack((final_image, final_image))  # Дополнение снизу еще раз
    
    return final_image

def process_images(input_folder, output_folder):
    """
    Обрабатывает все изображения в указанной папке и сохраняет результат.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            processed_image = process_image(input_path)
            if processed_image is not None:
                cv2.imwrite(output_path, processed_image)
                print(f"Сохранено: {output_path}")

if __name__ == "__main__":
    # Пример использования
    input_folder = r"C:\Users\alexe\VSCprojects\lidar-visual-interface\temp\img_rgb"  # Укажите папку с изображениями
    output_folder = r"C:\Users\alexe\VSCprojects\lidar-visual-interface\temp\img_rgb_joined"  # Укажите папку для сохранения
    process_images(input_folder, output_folder)
