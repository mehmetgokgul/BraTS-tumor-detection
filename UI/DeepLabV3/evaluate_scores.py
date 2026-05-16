import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm  # İlerleme çubuğu (yüklü değilse: pip install tqdm)

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\fmaci\OneDrive\Desktop\MICCAI_BraTS2020_TrainingData"
MODEL_PATH = "brats_deeplabv3_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    """
    Modelin sayısal başarısını (Dice Score) ölçen analiz sınıfı.
    """

    def __init__(self, root_dir, model_path, test_indices):
        self.root_dir = root_dir
        self.device = DEVICE
        self.test_indices = test_indices
        self.patient_folders = sorted(glob.glob(os.path.join(root_dir, "BraTS20_Training_*")))

        # Modeli Yükle
        print(f"Model yükleniyor: {model_path}")
        self.model = self._load_model(model_path)
        print("Model hazır! Test işlemi başlıyor...")

    def _load_model(self, path):
        # aux_loss=True önemli!
        model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
        model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print("HATA: Model dosyası bulunamadı!")
            exit()

        model.to(self.device)
        model.eval()
        return model

    def get_patient_data(self, idx):
        if idx >= len(self.patient_folders): return None, None, None, None

        path = self.patient_folders[idx]
        p_id = os.path.basename(path)

        # Dosyalar
        flair = nib.load(os.path.join(path, f"{p_id}_flair.nii")).get_fdata()
        t1ce = nib.load(os.path.join(path, f"{p_id}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(path, f"{p_id}_t2.nii")).get_fdata()

        # Seg dosyası (İsim kontrolü)
        seg_path = os.path.join(path, f"{p_id}_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = glob.glob(os.path.join(path, "*seg*.nii"))[0]
        seg = nib.load(seg_path).get_fdata()

        # Etiket Düzeltme (4 -> 3)
        seg[seg == 4] = 3

        return flair, t1ce, t2, seg

    def calculate_dice(self, pred_mask, true_mask, num_classes=4):
        """
        Tek bir kesit için Dice skorlarını hesaplar.
        Döndürdüğü: [Nekroz Skoru, Ödem Skoru, Aktif Tümör Skoru]
        """
        dice_scores = []

        # Class 0 (Arkaplan) hariç diğerlerine bakıyoruz (1, 2, 3)
        for class_idx in range(1, num_classes):
            p = (pred_mask == class_idx).astype(int)
            t = (true_mask == class_idx).astype(int)

            intersection = np.sum(p * t)
            union = np.sum(p) + np.sum(t)

            if union == 0:
                # İkisi de boşsa (Tümör yok ve model de bulmadıysa) tam puan
                score = 1.0
            else:
                score = (2.0 * intersection) / (union + 1e-8)

            dice_scores.append(score)

        return dice_scores  # [Dice_1, Dice_2, Dice_3]

    def run_evaluation(self):
        """
        Belirlenen tüm test hastalarını tarar ve rapor basar.
        """
        total_scores = np.zeros(3)  # [Toplam_Necrosis, Toplam_Edema, Toplam_Enhancing]
        count = 0

        print("\n" + "=" * 60)
        print(f"{'HASTA ID':<20} | {'NEKROZ':<10} | {'ÖDEM':<10} | {'AKTİF':<10} | {'ORTALAMA':<10}")
        print("-" * 60)

        # Sadece test için ayırdığımız 100 ile 120 arasındaki hastalara bak
        for idx in self.test_indices:
            try:
                flair, t1ce, t2, seg = self.get_patient_data(idx)
                if flair is None: continue

                # Tümörün en net olduğu kesiti bul (Adil karşılaştırma için)
                tumor_counts = np.sum(seg > 0, axis=(0, 1))
                best_slice = np.argmax(tumor_counts)

                if tumor_counts[best_slice] == 0:
                    best_slice = 75  # Tümör yoksa ortaya bak

                # Veriyi Hazırla
                img_slice = np.stack([flair[:, :, best_slice], t1ce[:, :, best_slice], t2[:, :, best_slice]], axis=0)
                img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Tahmin
                with torch.no_grad():
                    output = self.model(tensor)['out']
                    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                # Skor Hesapla
                true_mask_slice = seg[:, :, best_slice]
                scores = self.calculate_dice(pred_mask, true_mask_slice)
                mean_score = np.mean(scores)

                # Tabloya Yaz
                p_id = os.path.basename(self.patient_folders[idx])
                print(
                    f"{p_id:<20} | {scores[0]:.4f}     | {scores[1]:.4f}     | {scores[2]:.4f}     | {mean_score:.4f}")

                # Ortalamaya Ekle
                total_scores += np.array(scores)
                count += 1

            except Exception as e:
                print(f"Hata (Hasta {idx}): {e}")

        # --- GENEL RAPOR ---
        avg_scores = total_scores / count
        grand_mean = np.mean(avg_scores)

        print("=" * 60)
        print("\n--- SONUÇ RAPORU (Test Seti: 20 Hasta) ---")
        print(f"Ortalama Nekroz Başarısı (Class 1) : {avg_scores[0]:.4f}")
        print(f"Ortalama Ödem Başarısı   (Class 2) : {avg_scores[1]:.4f}")
        print(f"Ortalama Aktif Tümör     (Class 3) : {avg_scores[2]:.4f}")
        print("-" * 40)
        print(f"GENEL MODEL SKORU (Mean Dice)      : {grand_mean:.4f}")
        print("=" * 60)


# --- ANA PROGRAM ---
if __name__ == "__main__":
    # Test Indices: Eğitimde kullanmadığımız 100 ile 120 arasındaki hastalar
    test_range = range(100, 120)

    evaluator = ModelEvaluator(ROOT_DIR, MODEL_PATH, test_range)
    evaluator.run_evaluation()