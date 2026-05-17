import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm  # İlerleme çubuğu için (pip install tqdm gerekebilir)

# --- 1. AYARLAR ---
ROOT_DIR = r"C:\Users\fmaci\OneDrive\Desktop\MICCAI_BraTS2021_TrainingData"
BATCH_SIZE = 8  # 2D olduğu için artırabilirsin (GPU hafızana göre 8, 16, 32 dene)
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- FUL EĞİTİM MODU ---")
print(f"Cihaz: {DEVICE}")


# --- 2. GELİŞMİŞ DATASET SINIFI ---
class BraTSFullDataset(Dataset):
    def __init__(self, root_dir, indices, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []  # (dosya_yolu, slice_index) çiftlerini tutacak

        # Tüm hasta klasörlerini al
        all_patients = sorted(glob.glob(os.path.join(root_dir, "BraTS21_Training_*")))

        # Sadece belirtilen indekslerdeki hastaları seç (0-100 veya 100-120)
        self.selected_patients = [all_patients[i] for i in indices if i < len(all_patients)]

        print(f"[{mode.upper()}] Veri seti hazırlanıyor... ({len(self.selected_patients)} hasta taranacak)")

        # ÖN İŞLEME: Her hastanın hangi slice'ında tümör var?
        for patient_path in tqdm(self.selected_patients):
            patient_id = os.path.basename(patient_path)

            # Seg dosyasını bul (İsim hatası kontrolü ile)
            seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
            if not os.path.exists(seg_path):
                potential = glob.glob(os.path.join(patient_path, "*seg*.nii"))
                if potential: seg_path = potential[0]

            if os.path.exists(seg_path):
                # Sadece slice indekslerini belirlemek için maskeyi yüklüyoruz
                # (Hepsini RAM'e almamak için mmap kullanıyoruz veya hızlıca bakıyoruz)
                seg_vol = nib.load(seg_path).get_fdata()

                # Tümör içeren slice'ları bul (toplamı 0'dan büyük olanlar)
                # axis=(0,1) -> H ve W boyunca topla, geriye Depth (slice) kalsın
                tumor_slices = np.where(np.sum(seg_vol, axis=(0, 1)) > 0)[0]

                # Listeye ekle: (Hasta Yolu, Slice Numarası)
                for s_idx in tumor_slices:
                    # Eğitimde çok fazla veri olacağı için her slice'ı almak yerine
                    # aradan seçebiliriz (stride). Şimdilik hepsini alıyoruz.
                    self.samples.append((patient_path, s_idx))

        print(f"[{mode.upper()}] Toplam {len(self.samples)} adet tümörlü 2D kesit bulundu.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_path, slice_idx = self.samples[idx]
        patient_id = os.path.basename(patient_path)

        # Dosya yolları
        flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
        t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
        t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

        # Dosya bulamazsa alternatif ara
        if not os.path.exists(seg_path):
            potential = glob.glob(os.path.join(patient_path, "*seg*.nii"))
            if potential: seg_path = potential[0]

        # Veriyi Yükle
        # NOT: Her slice için tüm 3D'yi yüklemek yavaştır ama en güvenilir yöntemdir.
        flair = nib.load(flair_path).dataobj[..., slice_idx]
        t1ce = nib.load(t1ce_path).dataobj[..., slice_idx]
        t2 = nib.load(t2_path).dataobj[..., slice_idx]
        seg = nib.load(seg_path).dataobj[..., slice_idx]

        # NumPy array'e çevir
        flair = np.array(flair, dtype=np.float32)
        t1ce = np.array(t1ce, dtype=np.float32)
        t2 = np.array(t2, dtype=np.float32)
        seg = np.array(seg, dtype=np.longlong)

        # İstifleme (3 Kanal)
        img_slice = np.stack([flair, t1ce, t2], axis=0)

        # Normalizasyon
        max_val = np.max(img_slice)
        if max_val > 0:
            img_slice = (img_slice - np.min(img_slice)) / (max_val + 1e-8)

        # Tensor Çevrimi
        image = torch.tensor(img_slice, dtype=torch.float32)
        mask = torch.tensor(seg, dtype=torch.long)

        # Etiket Düzeltme (4 -> 3)
        mask[mask == 4] = 3

        return image, mask


# --- 3. MODEL (ASPP ve DILATION) ---
def get_deeplabv3_model(num_classes=4):
    # DeepLabV3 ResNet50 omurgası ile yükleniyor.
    # DeepLabV3 mimarisi doğası gereği "Atrous Spatial Pyramid Pooling" (ASPP) kullanır.
    # Bu modül, görüntüyü 3 farklı dilation (genişleme) oranıyla işler.
    # Varsayılan Torchvision ayarlarında bu oranlar [12, 24, 36] (output_stride=8) veya [6, 12, 18] (output_stride=16) şeklindedir.

    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

    # 4 Sınıf için çıkış katmanını değiştiriyoruz
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model


# --- 4. DICE SKOR HESAPLAYICI ---
def calculate_dice(pred, target, num_classes=4):
    dice_scores = []
    pred = torch.argmax(pred, dim=1)

    # Background (0) hariç diğer sınıflar (1, 2, 3) için skor hesapla
    for i in range(1, num_classes):
        p = (pred == i).float()
        t = (target == i).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()

        if union == 0:
            dice_scores.append(1.0)  # İkisi de boşsa tam puan
        else:
            dice_scores.append((2. * intersection / (union + 1e-8)).item())

    return np.mean(dice_scores)


# --- 5. EĞİTİM DÖNGÜSÜ ---
def main():
    # 1. Datasetleri Hazırla
    # Train: 0'dan 100'e kadar olan hastalar
    train_dataset = BraTSFullDataset(ROOT_DIR, indices=range(0, 100), mode='train')
    # Validation: 100'den 120'ye kadar olan hastalar
    val_dataset = BraTSFullDataset(ROOT_DIR, indices=range(100, 120), mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Model, Loss, Optimizer
    model = get_deeplabv3_model(num_classes=4).to(DEVICE)

    # Dengesiz veri için ağırlıklı loss kullanılabilir ama şimdilik standart CrossEntropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_score = 0.0

    print("\n--- Model Eğitimi Başlıyor ---")
    print(f"Eğitim Verisi: {len(train_dataset)} kesit")
    print(f"Test Verisi: {len(val_dataset)} kesit")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0

        # İlerleme çubuğu ile eğitim
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)['out']

                # Batch içindeki ortalama Dice skorunu al
                batch_dice = calculate_dice(outputs, masks)
                val_dice_score += batch_dice

        avg_val_dice = val_dice_score / len(val_loader)

        print(f"Epoch {epoch + 1} Sonuç -> Loss: {epoch_loss:.4f} | Val Dice Score: {avg_val_dice:.4f}")

        # En iyi modeli kaydet
        if avg_val_dice > best_val_score:
            best_val_score = avg_val_dice
            torch.save(model.state_dict(), "brats_deeplabv3_best.pth")
            print(f"--> Yeni en iyi skor! Model kaydedildi. ({best_val_score:.4f})")

    print("\nEğitim Tamamlandı.")


if __name__ == "__main__":
    # Windows'ta multiprocessing hatası almamak için gerekli
    main()