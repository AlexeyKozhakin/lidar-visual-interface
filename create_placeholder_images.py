from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image(filename, text, size=(800, 600), bg_color=(240, 240, 240), text_color=(100, 100, 100)):
    """Создает пустое изображение с текстом-заглушкой"""
    
    # Создаем изображение
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Пытаемся использовать системный шрифт
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Разбиваем текст на строки
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] < size[0] - 40:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Рисуем текст по центру
    y_position = (size[1] - len(lines) * 30) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_position = (size[0] - text_width) // 2
        draw.text((x_position, y_position), line, fill=text_color, font=font)
        y_position += 30
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Сохраняем изображение
    img.save(filename)
    print(f"Создано: {filename}")

# Список отсутствующих изображений
missing_images = [
    ("report_images/predictions/multiclass_segmentation_results.png", "Multi-class Segmentation Results\n(Image to be added)"),
    ("report_images/postprocessing/stitched_prediction.png", "Stitched Prediction Results\n(Image to be added)"),
    ("report_images/3d/colored_3d_points.png", "3D Colored Point Cloud\n(Image to be added)"),
    ("report_images/3d/3d_classification_visualization.png", "3D Classification Visualization\n(Image to be added)"),
    ("report_images/ui/polygon_generation_app.png", "Polygon Generation Application\n(Screenshot to be added)"),
    ("report_images/ui/3d_segmentation_app.png", "3D Segmentation Application\n(Screenshot to be added)"),
    ("report_images/ui/processing_progress.png", "Processing Progress Interface\n(Screenshot to be added)"),
    ("report_images/ui/result_preview.png", "Result Preview Interface\n(Screenshot to be added)"),
    ("report_images/training/training_curves.png", "Training Progress Curves\n(Image to be added)"),
    ("report_images/training/model_architecture.png", "U-Net Model Architecture\n(Image to be added)")
]

# Создаем все заглушки
for filename, text in missing_images:
    create_placeholder_image(filename, text)

print("\nВсе изображения-заглушки созданы!") 