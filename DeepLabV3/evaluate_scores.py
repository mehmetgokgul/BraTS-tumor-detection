import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # Daha şık grafikler için (pip install seaborn gerekebilir)

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\fmaci\OneDrive\Desktop\MICCAI_BraTS2021_TrainingData"
MODEL_PATH = "brats_deeplabv3_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "images"  # Grafikleri kaydedeceğimiz klasör

# Klasör yoksa oluştur
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class ModelEvaluator:
    """
    Modelin sayısal başarısını (Dice Score) ölçen ve grafik çizen analiz sınıfı.
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

        flair = nib.load(os.path.join(path, f"{p_id}_flair.nii")).get_fdata()
        t1ce = nib.load(os.path.join(path, f"{p_id}_t1ce.nii")).get_fdata()
        t2 = nib.load(os.path.join(path, f"{p_id}_t2.nii")).get_fdata()

        seg_path = os.path.join(path, f"{p_id}_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = glob.glob(os.path.join(path, "*seg*.nii"))[0]
        seg = nib.load(seg_path).get_fdata()

        seg[seg == 4] = 3
        return flair, t1ce, t2, seg

    def calculate_dice(self, pred_mask, true_mask, num_classes=4):
        dice_scores = []
        for class_idx in range(1, num_classes):
            p = (pred_mask == class_idx).astype(int)
            t = (true_mask == class_idx).astype(int)
            intersection = np.sum(p * t)
            union = np.sum(p) + np.sum(t)

            if union == 0:
                score = 1.0
            else:
                score = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(score)
        return dice_scores

    def run_evaluation(self):
        total_scores = np.zeros(3)
        count = 0

        # Grafik için verileri biriktireceğimiz listeler
        all_necrosis = []
        all_edema = []
        all_enhancing = []

        print("\n" + "=" * 60)
        print(f"{'HASTA ID':<20} | {'NEKROZ':<10} | {'ÖDEM':<10} | {'AKTİF':<10} | {'ORTALAMA':<10}")
        print("-" * 60)

        for idx in tqdm(self.test_indices, desc="Test Verileri Taranıyor"):
            try:
                flair, t1ce, t2, seg = self.get_patient_data(idx)
                if flair is None: continue

                tumor_counts = np.sum(seg > 0, axis=(0, 1))
                best_slice = np.argmax(tumor_counts)

                if tumor_counts[best_slice] == 0:
                    best_slice = 75

                img_slice = np.stack([flair[:, :, best_slice], t1ce[:, :, best_slice], t2[:, :, best_slice]], axis=0)
                img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(tensor)['out']
                    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                true_mask_slice = seg[:, :, best_slice]
                scores = self.calculate_dice(pred_mask, true_mask_slice)
                mean_score = np.mean(scores)

                # Listelere ekle
                all_necrosis.append(scores[0])
                all_edema.append(scores[1])
                all_enhancing.append(scores[2])

                p_id = os.path.basename(self.patient_folders[idx])
                # TQDM kullandığımız için terminal çıktısını bozmaması adına tqdm.write kullanıyoruz
                tqdm.write(
                    f"{p_id:<20} | {scores[0]:.4f}     | {scores[1]:.4f}     | {scores[2]:.4f}     | {mean_score:.4f}")

                total_scores += np.array(scores)
                count += 1

            except Exception as e:
                print(f"Hata (Hasta {idx}): {e}")

        # --- GENEL RAPOR ---
        avg_scores = total_scores / count
        grand_mean = np.mean(avg_scores)

        print("=" * 60)
        print("\n--- SONUÇ RAPORU (Test Seti) ---")
        print(f"Ortalama Nekroz Başarısı (Class 1) : {avg_scores[0]:.4f}")
        print(f"Ortalama Ödem Başarısı   (Class 2) : {avg_scores[1]:.4f}")
        print(f"Ortalama Aktif Tümör     (Class 3) : {avg_scores[2]:.4f}")
        print("-" * 40)
        print(f"GENEL MODEL SKORU (Mean Dice)      : {grand_mean:.4f}")
        print("=" * 60)

        # --- GRAFİKLERİ ÇİZ VE KAYDET ---
        self.plot_dice_scores(all_necrosis, all_edema, all_enhancing, avg_scores)

    def plot_dice_scores(self, necrosis, edema, enhancing, avg_scores):
        """Dice skorlarını Box Plot ve Bar Chart olarak görselleştirir."""
        print("Grafikler oluşturuluyor...")
        sns.set_theme(style="whitegrid")
        labels = ["Nekroz (Core)", "Ödem (Edema)", "Aktif (Enhancing)"]

        # 1. KUTU GRAFİĞİ (Box Plot) - Hastalar arası dağılımı gösterir
        plt.figure(figsize=(10, 6))
        data_to_plot = [necrosis, edema, enhancing]
        sns.boxplot(data=data_to_plot, palette="Set2")
        plt.xticks(ticks=[0, 1, 2], labels=labels, fontsize=12)
        plt.ylabel("Dice Skoru (0 - 1.0)", fontsize=12)
        plt.title("Test Verilerinde Tümör Sınıflarına Göre Dice Skoru Dağılımı", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.1)

        # Grafiği Kaydet
        boxplot_path = os.path.join(SAVE_DIR, "dice_boxplot.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Kutu Grafiği kaydedildi: {boxplot_path}")

        # 2. ORTALAMA ÇUBUK GRAFİĞİ (Bar Chart)
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=labels, y=avg_scores, palette="viridis")
        plt.ylabel("Ortalama Dice Skoru", fontsize=12)
        plt.title("Modelin Ortalama Başarı Oranları (Test Seti)", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.0)

        # Çubukların üstüne sayısal değerleri yaz
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontweight='bold')

        # Grafiği Kaydet
        barplot_path = os.path.join(SAVE_DIR, "dice.png")
        plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Çubuk Grafiği kaydedildi: {barplot_path}")


if __name__ == "__main__":
    test_range = range(100, 120)
    evaluator = ModelEvaluator(ROOT_DIR, MODEL_PATH, test_range)
    evaluator.run_evaluation()