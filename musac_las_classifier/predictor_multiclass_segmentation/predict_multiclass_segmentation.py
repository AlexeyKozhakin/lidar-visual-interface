import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# === Классы и цвета ===
class_to_color = {
    0: [0, 0, 0],
    1: [180, 180, 180],
    2: [0, 255, 0],
    3: [255, 255, 0],
    4: [255, 0, 0],
    5: [135, 206, 250],
    6: [135, 206, 251],
    7: [135, 206, 252],
    8: [135, 206, 253],
    9: [135, 206, 254],
    10: [0, 0, 1],
    11: [0, 0, 2],
    12: [0, 0, 3],
    13: [190, 153, 153],
    14: [190, 153, 154],
    15: [0, 0, 4],
    16: [0, 0, 5],
    17: [180, 180, 181],
    18: [0, 0, 6],
    19: [0, 254, 0],
}

def class_to_rgb(class_mask, class_to_color):
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_to_color.items():
        rgb_mask[class_mask == class_idx] = color
    return rgb_mask

# === Dataset для предсказания ===
class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_filenames = sorted(self.image_dir.glob("*.png"))  # Или другое расширение

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path.name

# === Загрузка модели ===
def load_model(checkpoint_path, num_classes, device="cpu"):
    # Путь к весам encoder (если нужно вручную загружать)
    encoder_weights_path = "predictor_multiclass_segmentation/model/resnet34-333f7ec4.pth"

    # Загружаем веса encoder
    encoder_state_dict = torch.load(encoder_weights_path, weights_only=False)

    # Создаем модель без автоматической загрузки весов
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # чтобы не скачивались веса с интернета
        in_channels=3,
        classes=num_classes
    )

    # Загружаем encoder вручную
    model.encoder.load_state_dict(encoder_state_dict)

    # Загружаем checkpoint всей модели
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.to(device)
    model.eval()

    return model

# === Предсказание и сохранение ===
def predict_and_save(model, dataloader, save_dir, class_to_color, device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)  # (B, C, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # (B, H, W)

            for pred, filename in zip(preds, filenames):
                rgb_mask = class_to_rgb(pred, class_to_color)
                pred_pil = Image.fromarray(rgb_mask)
                pred_pil.save(os.path.join(save_dir, filename))

# === Основная функция ===
def main_prediction(input_directory, output_directory, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(class_to_color)

    transform = T.Compose([
        T.ToTensor(),
        # Если нужно добавить Resize или Normalize — здесь
    ])

    dataset = PredictionDataset(input_directory, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = load_model(checkpoint_path, num_classes, device)
    predict_and_save(model, dataloader, output_directory, class_to_color, device)

if __name__ == "__main__":
    input_directory = r"temp/img_features_join"
    checkpoint_path = r"predictor_multiclass_segmentation\model\model_epoch_31.pth"
    output_directory = r"temp/img_features_join_multi_class"

    main_prediction(input_directory, output_directory, checkpoint_path)
