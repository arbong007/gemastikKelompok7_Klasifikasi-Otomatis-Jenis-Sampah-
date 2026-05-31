import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn as nn

BASE_DIR = '../dataset/Garbage classification'
OUTPUT_DIR = '../output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_lengkap = datasets.ImageFolder(root=BASE_DIR, transform=transform_val)
ukuran_train = int(0.8 * len(dataset_lengkap))
ukuran_val = len(dataset_lengkap) - ukuran_train
torch.manual_seed(42)
_, dataset_val = random_split(dataset_lengkap, [ukuran_train, ukuran_val])
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)

# Load Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('../models/pytorch_garbage_model.pth'))
model.eval()

label_asli = []
prediksi_label = []
prediksi_skor = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        skor = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        label_asli.extend(labels.numpy())
        prediksi_label.extend(preds.numpy())
        prediksi_skor.extend(skor.numpy())

label_asli = np.array(label_asli)
prediksi_label = np.array(prediksi_label)
prediksi_skor = np.array(prediksi_skor)
nama_kelas = dataset_lengkap.classes

# VISUALISASI 1: Confusion Matrix Heatmap (Sangat disukai Juri)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(label_asli, prediksi_label)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nama_kelas, yticklabels=nama_kelas)
plt.title('Confusion Matrix - Distribusi Misklasifikasi Sampah')
plt.ylabel('Kategori Asli')
plt.xlabel('Kategori Prediksi Sistem')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

# VISUALISASI 2: Grafik Batang Multi-Metrik Per Kelas
report_dict = classification_report(label_asli, prediksi_label, target_names=nama_kelas, output_dict=True)
kelas_plot = [k for k in nama_kelas]
precision_vals = [report_dict[k]['precision'] for k in nama_kelas]
recall_vals = [report_dict[k]['recall'] for k in nama_kelas]

x = np.arange(len(kelas_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, precision_vals, width, label='Precision', color='#1f77b4')
ax.bar(x + width/2, recall_vals, width, label='Recall', color='#ff7f0e')
ax.set_title('Perbandingan Validasi Precision vs Recall Per Kategori Sampah')
ax.set_xticks(x)
ax.set_xticklabels(kelas_plot)
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performa_metrik_batang.png'), dpi=300)
plt.close()

# VISUALISASI 3: Kurva ROC (Receiver Operating Characteristic) & AUC Score
label_asli_bin = label_binarize(label_asli, classes=[0, 1, 2, 3, 4, 5])
plt.figure(figsize=(8, 6))
for i in range(6):
    fpr, tpr, _ = roc_curve(label_asli_bin[:, i], prediksi_skor[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Kurva {nama_kelas[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
plt.title('Kurva Multikelas ROC-AUC - Reliabilitas Distribusi Data')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_auc_curve.png'), dpi=300)
plt.close()

print("Eksperimen Sukses! Periksa folder '../output/' untuk mengambil 3 grafik analisis laporan baru Anda.")