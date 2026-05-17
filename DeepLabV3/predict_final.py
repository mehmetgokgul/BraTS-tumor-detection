import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- AYARLAR ---
ROOT_DIR = r"C:\Users\fmaci\OneDrive\Desktop\MICCAI_BraTS2021_TrainingData"
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
        self.patient_folders = sorted(glob.glob(os.path.join(root_dir, "BraTS*")))

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
        """Belirtilen indeksteki hastanın verilerini diskten okur (Eski ve Yeni Format Uyumlu)."""
        if idx >= len(self.patient_folders): return None, None, None, None, None

        path = self.patient_folders[idx]
        p_id = os.path.basename(path)

        # 1. FLAIR / t2f
        flair_path = os.path.join(path, f"{p_id}_flair.nii")
        if not os.path.exists(flair_path):  # Eğer eski format yoksa, yeni formata bak
            flair_path = os.path.join(path, f"{p_id}-t2f.nii")
        flair = nib.load(flair_path).get_fdata()

        # 2. T1CE / t1c
        t1ce_path = os.path.join(path, f"{p_id}_t1ce.nii")
        if not os.path.exists(t1ce_path):
            t1ce_path = os.path.join(path, f"{p_id}-t1c.nii")
        t1ce = nib.load(t1ce_path).get_fdata()

        # 3. T2 / t2w
        t2_path = os.path.join(path, f"{p_id}_t2.nii")
        if not os.path.exists(t2_path):
            t2_path = os.path.join(path, f"{p_id}-t2w.nii")
        t2 = nib.load(t2_path).get_fdata()

        # 4. SEGMENTASYON
        seg_path = os.path.join(path, f"{p_id}_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(path, f"{p_id}-seg.nii")
        if not os.path.exists(seg_path): # Tolerans
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

   # def show_comparison(self, img, true_mask, pred_mask, patient_id, slice_idx):
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

    def interactive_show(self, flair, t1ce, t2, true_mask, patient_id):
        """
        Slider ile 3D hacim üzerinde katman katman (slice by slice) gezinmeyi sağlar.
        """
        true_mask[true_mask == 4] = 3

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.25)  # Slider için alt tarafta boşluk bırak

        # Başlangıç kesitini tümörün en yoğun olduğu yer olarak belirle
        tumor_counts = np.sum(true_mask > 0, axis=(0, 1))
        init_slice = np.argmax(tumor_counts)
        if tumor_counts[init_slice] == 0:
            init_slice = flair.shape[2] // 2

        # İlk tahmini al
        img_show, pred_mask = self.predict_slice(flair, t1ce, t2, init_slice)

        # İlk Görüntüleri Çiz
        im1 = ax1.imshow(img_show, cmap='gray')
        ax1.set_title(f"MRI (Flair)", fontsize=14)
        ax1.axis('off')

        im2 = ax2.imshow(true_mask[:, :, init_slice], cmap='jet', vmin=0, vmax=3)
        ax2.set_title("Doktorun İşaretlediği", fontsize=14, color='green')
        ax2.axis('off')

        im3 = ax3.imshow(pred_mask, cmap='jet', vmin=0, vmax=3)
        ax3.set_title("DeepLabV3 Tahmini", fontsize=14, color='blue')
        ax3.axis('off')

        fig.suptitle(f"HASTA: {patient_id} - Kesit: {init_slice}", fontsize=16, weight='bold')

        # Slider'ı Ekle
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
        slider = Slider(
            ax=ax_slider,
            label='Kesit (Slice)',
            valmin=0,
            valmax=flair.shape[2] - 1,
            valinit=init_slice,
            valstep=1
        )

        # Slider hareket ettiğinde çalışacak dinamik fonksiyon
        def update(val):
            s_idx = int(slider.val)
            # O anki kesit için anlık tahmin yap
            img_slice, p_mask = self.predict_slice(flair, t1ce, t2, s_idx)

            # Ekrandaki resimleri güncelle
            im1.set_data(img_slice)
            im2.set_data(true_mask[:, :, s_idx])
            im3.set_data(p_mask)

            # Başlıktaki kesit numarasını güncelle
            fig.suptitle(f"HASTA: {patient_id} - Kesit: {s_idx}", fontsize=16, weight='bold')
            fig.canvas.draw_idle()

        # Slider değişimi dinleniyor
        slider.on_changed(update)
        plt.show()



if __name__ == "__main__":
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

            print("Görselleştirme arayüzü başlatılıyor. Alt kısımdaki slider ile kesitleri gezebilirsiniz.")

            # Slider arayüzünü başlat
            predictor.interactive_show(flair, t1ce, t2, seg, p_id)

        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
        except Exception as e:
            print(f"Bir hata oluştu: {e}")