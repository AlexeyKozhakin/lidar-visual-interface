import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from torchvision.models import resnet34

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


import torch
import segmentation_models_pytorch as smp

def load_model(checkpoint_path, device="cpu"):
    # Путь к весам encoder
    encoder_weights_path = "predictor_building_segmentation/model/resnet34-333f7ec4.pth"
    
    # Загружаем веса encoder из .pth-файла (обязательно: weights_only=False)
    encoder_state_dict = torch.load(encoder_weights_path, weights_only=False)

    # Создаем модель с отключенной автоматической загрузкой весов
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # чтобы не скачивались веса с интернета
        in_channels=3,
        classes=2
    )

    # Загружаем encoder вручную
    model.encoder.load_state_dict(encoder_state_dict)

    # Загружаем обученные веса всей модели (Unet) из чекпойнта
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Отправляем на устройство и в режим оценки
    model.to(device)
    model.eval()

    return model



def predict_and_save(model, dataloader, save_dir, device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)  # Предсказание
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Берем класс с наибольшей вероятностью
            
            for pred, filename in zip(preds, filenames):
                pred_image = (pred * 255).astype(np.uint8)  # Масштабируем в [0, 255]
                pred_pil = Image.fromarray(pred_image, mode="L")  # Ч/б изображение
                pred_pil.save(os.path.join(save_dir, filename))

def main_prediction(input_directory, output_directory, checkpoint_path):
    transform = T.Compose([
        T.ToTensor(),
        #T.Resize((64, 64))
    ])
    
    dataset = PredictionDataset(input_directory, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    model = load_model(checkpoint_path)
    predict_and_save(model, dataloader, output_directory)

if __name__ == "__main__":

    input_directory = "temp/img_features"
    checkpoint_path = "predictor_building_segmentation/model/model_epoch_25.pth"
    output_directory = "temp/img_predict"
    
    main_prediction(input_directory, output_directory, checkpoint_path)

