import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

DATA_DIR = "data/diabetic_retinopathy/gaussian_filtered_images/gaussian_filtered_images"
CSV_FILE = "data/test_set.csv"
MODEL_FILE = f"models/resnet18_{__import__('sys').argv[1]}.pth"

BATCH_SIZE = 16

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", DEVICE)

class RetinopathyDataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_id = row["id_code"]
        label = int(row["diagnosis"])

        if label == 0:
            class_name = "No_DR"
        else:
            class_name = "Mild"

        image_path = os.path.join(
            self.data_dir,
            class_name,
            image_id + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = RetinopathyDataset(
    CSV_FILE,
    DATA_DIR,
    transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test images:", len(dataset))

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

correct = 0
total = 0

class_correct = {0: 0, 1: 0}
class_total = {0: 0, 1: 0}

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        for label, prediction in zip(labels, predictions):

            label = label.item()
            prediction = prediction.item()

            class_total[label] += 1

            if label == prediction:
                class_correct[label] += 1


accuracy = 100 * correct / total

print()
print("========== CLEAN TEST RESULTS ==========")
print(f"Total test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Overall accuracy: {accuracy:.2f}%")

print()
print("Class 0 (No_DR):")
print(
    f"{class_correct[0]}/{class_total[0]} "
    f"= {100 * class_correct[0] / class_total[0]:.2f}%"
)

print()
print("Class 1 (Mild):")
print(
    f"{class_correct[1]}/{class_total[1]} "
    f"= {100 * class_correct[1] / class_total[1]:.2f}%"
)
