import os
import sys
import random
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Please specify a dataset:")
    print("50_50, 70_30, or 80_20")
    sys.exit()

DISTRIBUTION = sys.argv[1]

DATA_DIR = "data/diabetic_retinopathy/gaussian_filtered_images/gaussian_filtered_images"
CSV_FILE = f"data/experiments/train_{DISTRIBUTION}.csv"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

SEED = 123

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", DEVICE)
print("Surrogate distribution:", DISTRIBUTION)
print("Random seed:", SEED)

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
    shuffle=True
)

print("Training images:", len(dataset))

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress.set_postfix(
            loss=loss.item(),
            accuracy=100 * correct / total
        )

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_accuracy:.2f}%"
    )

os.makedirs("models", exist_ok=True)

model_path = f"models/surrogate_resnet18_{DISTRIBUTION}.pth"

torch.save(
    model.state_dict(),
    model_path
)

print()
print("Surrogate model saved to:")
print(model_path)