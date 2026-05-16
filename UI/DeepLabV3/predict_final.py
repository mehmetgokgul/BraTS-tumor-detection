import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\fmaci\OneDrive\Desktop\MICCAI_BraTS2020_TrainingData"
MODEL_PATH = "brats_deeplabv3_best.pth"  # Eğitimden çıkan dosya
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TumorPredictor:
    """
    Bu sınıf, eğitilmiş DeepLabV3 modelini yükler ve
    istenen hastanın görüntüleri üzerinde tahmin yapar.
    """

    def __init__(self, model_path, root_dir):
        self.root_dir = root_dir
        self.device = DEVICE
        self.patient_folders = sorted(glob.glob(os.path.join(root_dir, "BraTS20_Training_*")))

        # Modeli Yükle
        print(f"Model yükleniyor: {model_path}")
        self.model = self._load_model(model_path)
        print("Model hazır!")

    def _load_model(self, path):
        # DÜZELTME BURADA: aux_loss=True ekledik.
        # Bu sayede model 'aux_classifier' katmanını oluşturur ve hata almayız.
        model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)

        # Çıkış katmanlarını 4 sınıf (0,1,2,3) için ayarla
        model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
        # Artık aux_classifier "None" olmadığı için bu satır hata vermeyecek:
        model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))

        # Eğitilmiş ağırlıkları yükle
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(f"HATA: '{path}' dosyası bulunamadı! Lütfen dosya adını kontrol et.")
            exit()

        model.to(self.device)
        model.eval()  # Tahmin modu
        return model

    def get_patient(self, idx):
        """Belirtilen indeksteki hastanın verilerini diskten okur."""
        if idx >= len(self.patient_folders): return None, None, None, None, None

        path = self.patient_folders[idx]
        p_id = os.path.basename(path)

        # Dosya yolları
        flair = nib.load(os.path.join(path, f"{p_id}_flair.nii")).get_fdata()
        t1ce = nib.load(os.path.join(path, f"{p_id}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(path, f"{p_id}_t2.nii")).get_fdata()

        # Seg dosyasını bul (isim hatası toleransı)
        seg_path = os.path.join(path, f"{p_id}_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = glob.glob(os.path.join(path, "*seg*.nii"))[0]
        seg = nib.load(seg_path).get_fdata()

        return flair, t1ce, t2, seg, p_id

    def predict_slice(self, flair, t1ce, t2, slice_idx):
        """Tek bir kesit (slice) için tahmin üretir."""

        # Veri Hazırlığı (Stack & Normalize)
        img_slice = np.stack([
            flair[:, :, slice_idx],
            t1ce[:, :, slice_idx],
            t2[:, :, slice_idx]
        ], axis=0)

        # Normalizasyon (0-1 arası)
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

        # Tensor'a çevir
        tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Tahmin yap
        with torch.no_grad():
            output = self.model(tensor)['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        return img_slice[0], pred_mask  # Geriye orijinal resim (flair) ve tahmini döndür

    def show_comparison(self, img, true_mask, pred_mask, patient_id, slice_idx):
        """
        Üçlü karşılaştırma yapar:
        1. Orijinal MRI
        2. Gerçek (Doktor) Maskesi
        3. Yapay Zeka Tahmini
        """
        # Maske etiket düzeltmesi (görselleştirme için 4'ü 3 yapalım)
        true_mask[true_mask == 4] = 3

        plt.figure(figsize=(18, 6))

        # 1. MRI Görüntüsü
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"MRI (Flair) - Kesit: {slice_idx}", fontsize=14)
        plt.axis('off')

        # 2. Gerçek Maske
        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='jet', vmin=0, vmax=3)
        plt.title("Doktorun İşaretlediği (Ground Truth)", fontsize=14, color='green')
        plt.axis('off')

        # 3. Model Tahmini
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=3)
        plt.title("DeepLabV3 Tahmini (Bizim Model)", fontsize=14, color='blue')
        plt.axis('off')

        plt.suptitle(f"HASTA: {patient_id}", fontsize=16, weight='bold')
        plt.show()


# --- ANA PROGRAM ---
if __name__ == "__main__":
    # Sınıfı Başlat
    predictor = TumorPredictor(MODEL_PATH, ROOT_DIR)

    while True:
        try:
            val = input(
                f"\nİncelenecek hasta numarasını gir (0 - {len(predictor.patient_folders) - 1}) (Çıkış için 'q'): ")
            if val.lower() == 'q':
                break

            idx = int(val)
            print(f"{idx}. hasta verisi yükleniyor...")

            # Verileri Çek
            flair, t1ce, t2, seg, p_id = predictor.get_patient(idx)
            if flair is None:
                print("Hatalı numara!")
                continue

            # Otomatik en iyi kesiti bul (Tümörün en büyük olduğu yer)
            print("Tümör taranıyor...")
            tumor_counts = np.sum(seg > 0, axis=(0, 1))
            best_slice = np.argmax(tumor_counts)

            # Eğer hiç tümör yoksa ortaya bak
            if tumor_counts[best_slice] == 0:
                print("UYARI: Bu hastada belirgin tümör kaydı yok, orta kesit gösteriliyor.")
                best_slice = 75
            else:
                print(f"Tümör tespit edildi! En net görüldüğü kesit: {best_slice}")

            # Tahmin Yap
            img_show, pred_mask = predictor.predict_slice(flair, t1ce, t2, best_slice)

            # Sonucu Göster
            predictor.show_comparison(img_show, seg[:, :, best_slice], pred_mask, p_id, best_slice)

        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")