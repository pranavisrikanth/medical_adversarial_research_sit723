import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

parser = argparse.ArgumentParser(description="FGSM black-box transfer attack")

parser.add_argument(
    "--surrogate",
    required=True,
    choices=["50_50", "70_30", "80_20"],
    help="Surrogate model class distribution"
)

parser.add_argument(
    "--target",
    required=True,
    choices=["50_50", "70_30", "80_20"],
    help="Target model class distribution"
)

args = parser.parse_args()

SURROGATE_DISTRIBUTION = args.surrogate
TARGET_DISTRIBUTION = args.target

SURROGATE_FILE = f"models/surrogate_resnet18_{SURROGATE_DISTRIBUTION}.pth"
TARGET_FILE = f"models/resnet18_{TARGET_DISTRIBUTION}.pth"

DATA_DIR = "data/diabetic_retinopathy/gaussian_filtered_images/gaussian_filtered_images"
CSV_FILE = "data/test_set.csv"

BATCH_SIZE = 16
EPSILON = 0.01

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", DEVICE)
print("FGSM epsilon:", EPSILON)
print("Surrogate distribution:", SURROGATE_DISTRIBUTION)
print("Target distribution:", TARGET_DISTRIBUTION)


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


def create_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model = model.to(DEVICE)

    return model


surrogate = create_model()

surrogate.load_state_dict(
    torch.load(
        SURROGATE_FILE,
        map_location=DEVICE
    )
)

surrogate.eval()


target = create_model()

target.load_state_dict(
    torch.load(
        TARGET_FILE,
        map_location=DEVICE
    )
)

target.eval()


def fgsm_attack(model, images, labels, epsilon):

    images = images.clone().detach()

    images.requires_grad = True

    outputs = model(images)

    loss = nn.CrossEntropyLoss()(
        outputs,
        labels
    )

    model.zero_grad()

    loss.backward()

    gradient = images.grad.data

    perturbed_images = (
        images + epsilon * gradient.sign()
    )

    perturbed_images = perturbed_images.detach()

    return perturbed_images


clean_correct = 0
target_correct_on_adv = 0
successful_transfers = 0
total = 0


for images, labels in loader:

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():

        surrogate_outputs = surrogate(images)

        surrogate_predictions = torch.argmax(
            surrogate_outputs,
            dim=1
        )

    correctly_classified = (
        surrogate_predictions == labels
    )

    if correctly_classified.sum().item() == 0:
        continue

    attack_images = images[
        correctly_classified
    ]

    attack_labels = labels[
        correctly_classified
    ]

    adversarial_images = fgsm_attack(
        surrogate,
        attack_images,
        attack_labels,
        EPSILON
    )

    with torch.no_grad():

        target_outputs = target(
            adversarial_images
        )

        target_predictions = torch.argmax(
            target_outputs,
            dim=1
        )

    clean_correct += attack_labels.size(0)

    target_correct_on_adv += (
        target_predictions == attack_labels
    ).sum().item()

    successful_transfers += (
        target_predictions != attack_labels
    ).sum().item()

    total += attack_labels.size(0)


if total > 0:

    transfer_success_rate = (
        100 * successful_transfers / total
    )

    target_adv_accuracy = (
        100 * target_correct_on_adv / total
    )

    print()
    print("========== FGSM TRANSFER RESULTS ==========")

    print(
        f"Surrogate distribution: "
        f"{SURROGATE_DISTRIBUTION.replace('_', ':')}"
    )

    print(
        f"Target distribution: "
        f"{TARGET_DISTRIBUTION.replace('_', ':')}"
    )

    print(
        f"Surrogate correctly classified: "
        f"{clean_correct}/{total}"
    )

    print(
        f"Target correct on adversarial images: "
        f"{target_correct_on_adv}/{total}"
    )

    print(
        f"Successful transfers: "
        f"{successful_transfers}/{total}"
    )

    print(
        f"Transfer attack success rate: "
        f"{transfer_success_rate:.2f}%"
    )

    print(
        f"Target adversarial accuracy: "
        f"{target_adv_accuracy:.2f}%"
    )

else:

    print(
        "No correctly classified images available for attack."
    )