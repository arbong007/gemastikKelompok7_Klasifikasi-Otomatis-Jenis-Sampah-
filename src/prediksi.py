import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('../models/pytorch_garbage_model.pth'))
model.eval()

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
transform_uji = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prediksi_sampah_warga(img_path):
    if not os.path.exists(img_path):
        print(f"Berkas gambar tidak ditemukan: {img_path}")
        return
    img = Image.open(img_path).convert('RGB')
    img_t = transform_uji(img)
    img_b = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        outputs = model(img_b)
        probabilitas = torch.nn.functional.softmax(outputs[0], dim=0)
        indeks_terpilih = torch.argmax(probabilitas).item()
        
    print(f"\n" + "="*40)
    print(f" DEMO INFERENSI VALIDASI REAL-TIME ")
    print(f"="*40)
    print(f"Nama File Citra : {os.path.basename(img_path)}")
    print(f"Hasil Klasifikasi: [{class_labels[indeks_terpilih].upper()}]")
    print(f"Nilai Confidence : {probabilitas[indeks_terpilih].item() * 100:.2f}%")
    print(f"="*40 + "\n")

if __name__ == "__main__":
    prediksi_sampah_warga('foto_uji.jpg')