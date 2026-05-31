import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Inisialisasi Jalur Folder
BASE_DIR = '../dataset/Garbage classification'
MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Transformasi & Augmentasi Gambar (Standard Komunitas Lapangan)
transformasi = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_lengkap = datasets.ImageFolder(root=BASE_DIR, transform=transformasi)
ukuran_train = int(0.8 * len(dataset_lengkap))
ukuran_val = len(dataset_lengkap) - ukuran_train
dataset_train, dataset_val = random_split(dataset_lengkap, [ukuran_train, ukuran_val])

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

# Model ResNet18 Pretrained (Transfer Learning)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Memulai pelatihan model standar GEMASTIK...")
model.train()
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/10] -> Loss: {running_loss / len(train_loader):.4f}")

# Menyimpan Model Resmi
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'pytorch_garbage_model.pth'))
print("Model PyTorch berhasil disimpan di folder 'models/'.")