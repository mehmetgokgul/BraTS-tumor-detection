import os
import sys
import nibabel as nib
import numpy as np
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp
 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ──────────────────────────────────────────────────────────────
#  UNet mimarisi — UNet/ klasöründen import
# ──────────────────────────────────────────────────────────────
UNET_AVAILABLE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNet"))
    from  Convolution import Convolution   # noqa: F401
    from Decoder import Decoder           # noqa: F401
    from UNet_Model import MyUNet
    UNET_AVAILABLE = True
    print("[OK] UNet modülleri yüklendi.")
except Exception as e:
    print(f"[UYARI] UNet modülleri yüklenemedi: {e}")


class NeuroScanApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("NeuroScan AI | Gelişmiş Tıbbi Raporlama")
        self.geometry("1500x950")

        # U-Net artık 4 kanal kullanıyor
        self.mri_data = {"FLAIR": None, "T1": None, "T1ce": None, "T2": None}
        self.gt_data  = None
        self.current_file_name   = "Bilinmeyen Dosya"
        self.current_model       = None
        self.selected_model_name = "U-Net"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Cihaz: {self.device}")

        self.multi_color_cmap = ListedColormap(['none', '#FF0000', '#00FF00', '#0000FF'])

        self.setup_ui()
        threading.Thread(target=self.load_selected_model, daemon=True).start()

    # ══════════════════════════════════════════════
    #  UI KURULUMU
    # ══════════════════════════════════════════════
    def setup_ui(self):
        self.left_panel = ctk.CTkFrame(self, width=370, corner_radius=0, fg_color="#1A1A1B")
        self.left_panel.pack(side="left", fill="y")
        ctk.CTkLabel(
            self.left_panel, text="NEUROSCAN AI",
            font=("Arial", 26, "bold"), text_color="#3B8ED0"
        ).pack(pady=(30, 10))

        self.model_menu = ctk.CTkOptionMenu(
            self.left_panel,
            values=["U-Net", "DeepLabV3", "EfficientNet"],
            command=self.change_model_event,
            width=280
        )
        self.model_menu.pack(pady=10)

        self.model_info_label = ctk.CTkLabel(
            self.left_panel, text="",
            font=("Arial", 10), text_color="gray", wraplength=320
        )
        self.model_info_label.pack(pady=(0, 5))

        self.zones = {}
        self.zones["FLAIR"] = self.create_drop_zone("1. FLAIR (t2f.nii)",      "#252526", "FLAIR")
        self.zones["T1"]    = self.create_drop_zone("2. T1n   (t1n.nii)",       "#252526", "T1")
        self.zones["T1ce"]  = self.create_drop_zone("3. T1c   (t1c.nii)",       "#252526", "T1ce")
        self.zones["T2"]    = self.create_drop_zone("4. T2w   (t2w.nii)",       "#252526", "T2")
        self.zones["mask"]  = self.create_drop_zone("5. MASKE (İsteğe Bağlı)", "#1e2a1e", "mask")

        self._update_model_info()

        self.perc_label = ctk.CTkLabel(
            self.left_panel, text="%0 Tamamlandı", font=("Arial", 14, "bold")
        )
        self.perc_label.pack(pady=(20, 0))
        self.progress_bar = ctk.CTkProgressBar(self.left_panel, width=300)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(self.left_panel, text="Model yükleniyor...")
        self.progress_label.pack()

        self.btn_run = ctk.CTkButton(
            self.left_panel, text="ANALİZİ BAŞLAT",
            command=self.start_analysis, state="disabled",
            font=("Arial", 16, "bold"), height=50
        )
        self.btn_run.pack(pady=30, padx=30, fill="x")

        self.right_panel = ctk.CTkScrollableFrame(self, fg_color="#0F0F0F", corner_radius=0)
        self.right_panel.pack(side="right", fill="both", expand=True)

    def _update_model_info(self):
        info = {
            "U-Net":        "Kanallar: FLAIR + T1n + T1c + T2w  ",
            "DeepLabV3":    "Kanallar: FLAIR + T1c + T2w        ",
            "EfficientNet": "Kanallar: FLAIR + T1n + T1c + T2w  ",
        }
        self.model_info_label.configure(text=info.get(self.selected_model_name, ""))

    def create_drop_zone(self, text, color, key):
        f = ctk.CTkFrame(self.left_panel, height=75, fg_color=color,
                         border_width=1, border_color="#333")
        f.pack(pady=5, padx=25, fill="x")
        l = ctk.CTkLabel(f, text=text, font=("Arial", 12))
        l.pack(expand=True, side="left", padx=15)
        btn_del = ctk.CTkButton(
            f, text="×", width=25, height=25, fg_color="#442222",
            command=lambda k=key: self.clear_zone(k)
        )
        btn_del.pack_forget()
        f.drop_target_register(DND_FILES)
        f.dnd_bind('<<Drop>>', lambda e, k=key, lbl=l, bd=btn_del: self.handle_drop(e, k, lbl, bd))
        return {"frame": f, "label": l, "del_btn": btn_del, "orig_text": text}

    # ══════════════════════════════════════════════
    #  HAZIR DURUM KONTROLÜ
    # ══════════════════════════════════════════════
    def check_ready_status(self):
        required = {
            "U-Net":        ["FLAIR", "T1", "T1ce", "T2"],
            "DeepLabV3":    ["FLAIR", "T1ce", "T2"],
            "EfficientNet": ["FLAIR", "T1", "T1ce", "T2"],
        }
        needed      = required.get(self.selected_model_name, ["FLAIR"])
        data_ready  = all(self.mri_data.get(k) is not None for k in needed)
        model_ready = self.current_model is not None

        if data_ready and model_ready:
            self.btn_run.configure(state="normal", fg_color="#1F6AA5")
        else:
            self.btn_run.configure(state="disabled", fg_color="#2D5A88")

    # ══════════════════════════════════════════════
    #  SÜRÜKLE-BIRAK
    # ══════════════════════════════════════════════
    def handle_drop(self, event, key, label, del_btn):
        path = event.data.strip('{}').strip('"')
        try:
            data = nib.load(path).get_fdata()
            if key == "mask":
                self.gt_data = data
            else:
                self.mri_data[key] = data
                if key == "FLAIR":
                    self.current_file_name = os.path.basename(path)
            label.configure(
                text=f"✓ {os.path.basename(path)[:20]}...", text_color="#4BB543"
            )
            del_btn.pack(side="right", padx=10)
            self.check_ready_status()
        except Exception as e:
            label.configure(text=f"Hata! {e}", text_color="red")

    def clear_zone(self, key):
        if key == "mask":
            self.gt_data = None
        else:
            self.mri_data[key] = None
        zone = self.zones[key]
        zone["label"].configure(text=zone["orig_text"], text_color="white")
        zone["del_btn"].pack_forget()
        self.check_ready_status()

    # ══════════════════════════════════════════════
    #  MODEL YÜKLEME + LAYER TESTİ
    # ══════════════════════════════════════════════
    def load_selected_model(self):
        self.current_model = None
        self.after(0, lambda: self.progress_label.configure(
            text=f"{self.selected_model_name} yükleniyor..."
        ))

        try:
            base = os.path.dirname(os.path.abspath(__file__))

            # ── U-Net (PyTorch, 4 kanal → 4 sınıf) ───────────────────────
            if self.selected_model_name == "U-Net":
                if not UNET_AVAILABLE:
                    raise ImportError(
                        "UNet modül dosyaları bulunamadı.\n"
                        "Convolution.py / Decoder.py / UNet_Model.py\n"
                        "UNet/ klasörünün içinde olmalı."
                    )

                w_path = os.path.join(base, "UNet", "UNet.pth")
                if not os.path.exists(w_path):
                    raise FileNotFoundError(f"Ağırlık dosyası bulunamadı:\n{w_path}")

                model = MyUNet(in_channels=4, num_classes=4)

                checkpoint = torch.load(w_path, map_location=self.device)

                # Checkpoint sarmalını aç
                if isinstance(checkpoint, dict):
                    state = (checkpoint.get('state_dict')
                             or checkpoint.get('model_state_dict')
                             or checkpoint.get('model')
                             or checkpoint)
                else:
                    state = checkpoint

                # "module." önekini temizle (DataParallel kayıtları için)
                cleaned = {k.replace("module.", ""): v for k, v in state.items()}

                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"[UYARI] Eksik anahtarlar ({len(missing)}): {missing[:3]} ...")
                if unexpected:
                    print(f"[UYARI] Beklenmeyen anahtarlar ({len(unexpected)}): {unexpected[:3]} ...")

                model.to(self.device).eval()
                self.current_model = model

                # Layer testi
                shape_info = self._test_unet(model)
                msg = f"U-Net Hazır ✓  "

            # ── DeepLabV3 (PyTorch, 3 kanal → 4 sınıf) ───────────────────
            elif self.selected_model_name == "DeepLabV3":
                model_path = os.path.join(base, "DeepLabV3", "DeepLabV3.pth")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model bulunamadı: {model_path}")
                pt_model = deeplabv3_resnet50(weights=None, aux_loss=True)
                pt_model.classifier[4]     = nn.Conv2d(256, 4, kernel_size=(1, 1))
                pt_model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1))
                checkpoint = torch.load(model_path, map_location=self.device)
                state = (checkpoint['state_dict']
                         if isinstance(checkpoint, dict) and 'state_dict' in checkpoint
                         else checkpoint)
                pt_model.load_state_dict(state)
                pt_model.to(self.device).eval()
                self.current_model = pt_model
                msg = "DeepLabV3 Hazır ✓"

            # ── EfficientNet (smp, 4 kanal → 3 sınıf) ────────────────────
            elif self.selected_model_name == "EfficientNet":
                model_path = os.path.join(base, "EfficientNet", "EfficientNet.pth")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model bulunamadı: {model_path}")
                eff_model = smp.Unet(
                    encoder_name="efficientnet-b0",
                    encoder_weights=None,
                    in_channels=4,
                    classes=3
                )
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = (checkpoint['state_dict']
                              if isinstance(checkpoint, dict) and 'state_dict' in checkpoint
                              else checkpoint)
                eff_model.load_state_dict(state_dict)
                eff_model.to(self.device).eval()
                self.current_model = eff_model
                msg = "EfficientNet Hazır ✓"

        except Exception as e:
            msg = f"HATA: {e}"
            print(f"[MODEL HATA] {e}")

        self.after(0, lambda: (
            self.progress_label.configure(text=msg),
            self.check_ready_status()
        ))

    # ── U-Net layer testi ─────────────────────────────────────────────────
    def _test_unet(self, model):
         
        hooks   = []
        results = {}

        def make_hook(name):
            def h(module, inp, out):
                results[name] = tuple(out.shape)
            return h

        layer_map = {
            "Encoder C1": model.c1,
            "Encoder C2": model.c2,
            "Encoder C3": model.c3,
            "Bottleneck":  model.bn,
            "Decoder D1":  model.d1,
            "Decoder D2":  model.d2,
            "Decoder D3":  model.d3,
            "Final Conv":  model.final,
        }
        for name, layer in layer_map.items():
            hooks.append(layer.register_forward_hook(make_hook(name)))

        dummy = torch.zeros(1, 4, 240, 240, device=self.device)
        with torch.no_grad():
            out = model(dummy)

        for h in hooks:
            h.remove()

        return f"Çıkış boyutu: {tuple(out.shape)}"

    def change_model_event(self, new_model):
        self.selected_model_name = new_model
        self._update_model_info()
        self.check_ready_status()
        threading.Thread(target=self.load_selected_model, daemon=True).start()

    # ══════════════════════════════════════════════
    #  UI KİLİT / AÇ
    # ══════════════════════════════════════════════
    def set_ui_state(self, state):
        self.model_menu.configure(state=state)
        for key in self.zones:
            zone = self.zones[key]
            if state == "disabled":
                zone["frame"].drop_target_unregister()
                zone["del_btn"].configure(state="disabled")
            else:
                zone["frame"].drop_target_register(DND_FILES)
                zone["del_btn"].configure(state="normal")

    # ══════════════════════════════════════════════
    #  ANALİZ BAŞLAT
    # ══════════════════════════════════════════════
    def start_analysis(self):
        self.btn_run.configure(state="disabled", text="ANALİZ EDİLİYOR...")
        self.set_ui_state("disabled")
        snap_mri        = {k: (v.copy() if v is not None else None) for k, v in self.mri_data.items()}
        snap_gt         = self.gt_data.copy() if self.gt_data is not None else None
        snap_name       = self.current_file_name
        # ▼▼▼ DÜZELTME: analiz başladığı andaki model adını yakala ▼▼▼
        snap_model_name = self.selected_model_name
        threading.Thread(
            target=self.process_and_display,
            args=(snap_mri, snap_gt, snap_name, snap_model_name),
            daemon=True
        ).start()

    # ══════════════════════════════════════════════
    #  YARDIMCI: Z-SCORE NORMALİZASYON
    # ══════════════════════════════════════════════
    def _z_score(self, arr):
        arr  = arr.astype(np.float32)
        mask = arr > 0
        if np.any(mask):
            arr[mask] = (arr[mask] - arr[mask].mean()) / (arr[mask].std() + 1e-8)
        return arr

    # ══════════════════════════════════════════════
    #  ANA İŞLEM DÖNGÜSÜ
    # ══════════════════════════════════════════════
    # ▼▼▼ DÜZELTME: snap_model_name parametresi eklendi ▼▼▼
    def process_and_display(self, snap_mri, snap_gt, snap_name, snap_model_name):
        flair       = snap_mri["FLAIR"]
        h, w, depth = flair.shape
        all_preds   = []

        for i in range(depth):
            f_slice = flair[:, :, i]

            # ── U-Net: 4 kanal → argmax → uint8 ──────────────────────────
            if snap_model_name == "U-Net":
                channels = [
                    self._z_score(f_slice),
                    self._z_score(snap_mri["T1"][:, :, i]),
                    self._z_score(snap_mri["T1ce"][:, :, i]),
                    self._z_score(snap_mri["T2"][:, :, i]),
                ]
                inp = torch.from_numpy(
                    np.stack(channels)           # (4, H, W)
                ).unsqueeze(0).float().to(self.device)   # (1, 4, H, W)

                with torch.no_grad():
                    out  = self.current_model(inp)         # (1, 4, H, W)
                    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                    # 0=arka plan  1=nekrotik  2=ödem  3=aktif tümör

            # ── EfficientNet: 4 kanal → sigmoid → multi-label ─────────────
            elif snap_model_name == "EfficientNet":
                channels = [
                    self._z_score(f_slice),
                    self._z_score(snap_mri["T1"][:, :, i]),
                    self._z_score(snap_mri["T1ce"][:, :, i]),
                    self._z_score(snap_mri["T2"][:, :, i]),
                ]
                inp = torch.from_numpy(
                    np.stack(channels)
                ).unsqueeze(0).float().to(self.device)

                with torch.no_grad():
                    out = torch.sigmoid(self.current_model(inp)).squeeze().cpu().numpy()
                    pred = np.zeros(out.shape[1:], dtype=np.uint8)
                    pred[out[0] > 0.5] = 1   # WT
                    pred[out[1] > 0.5] = 2   # TC
                    pred[out[2] > 0.5] = 3   # ET

            # ── DeepLabV3: 3 kanal → argmax ───────────────────────────────
            else:
                t1ce = (snap_mri["T1ce"][:, :, i]
                        if snap_mri["T1ce"] is not None else np.zeros_like(f_slice))
                t2   = (snap_mri["T2"][:, :, i]
                        if snap_mri["T2"]   is not None else np.zeros_like(f_slice))
                stacked = np.stack([f_slice, t1ce, t2], axis=0).astype(np.float32)
                mn, mx  = stacked.min(), stacked.max()
                norm    = (stacked - mn) / (mx - mn + 1e-8)
                inp     = torch.from_numpy(norm).unsqueeze(0).float().to(self.device)

                with torch.no_grad():
                    out  = self.current_model(inp)['out']
                    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Boyut garantisi (model farklı çözünürlükte çıkış verirse)
            if pred.shape != (h, w):
                pred_t = torch.from_numpy(
                    pred.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(
                    pred_t, size=(h, w), mode='nearest'
                ).squeeze().numpy().astype(np.uint8)

            all_preds.append(pred)

            # İlerleme çubuğu
            self.after(0, lambda p=int(((i + 1) / depth) * 100), v=(i + 1) / depth: (
                self.perc_label.configure(text=f"%{p} Tamamlandı"),
                self.progress_bar.set(v)
            ))

        # ▼▼▼ DÜZELTME: snap_model_name kartlara iletiliyor ▼▼▼
        self.after(0, lambda: (
            self.add_result_card(np.array(all_preds), flair, snap_gt, snap_name, snap_model_name),
            self.btn_run.configure(text="ANALİZİ BAŞLAT", state="normal"),
            self.set_ui_state("normal"),
            self.check_ready_status()
        ))

    # ══════════════════════════════════════════════
    #  SONUÇ KARTI
    # ══════════════════════════════════════════════
    # ▼▼▼ DÜZELTME: card_model_name parametresi eklendi, self.selected_model_name yerine kullanılıyor ▼▼▼
    def add_result_card(self, preds, local_flair, local_gt, file_name, card_model_name):
        depth = local_flair.shape[2]
        card  = ctk.CTkFrame(
            self.right_panel, fg_color="#161617",
            corner_radius=12, border_width=1, border_color="#333"
        )
        card.pack(pady=20, padx=15, fill="x")

        # Başlık — card_model_name sabit kalır, model değişse bile etkilenmez
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(pady=10, fill="x")
        ctk.CTkLabel(
            header,
            text=f"DOSYA: {file_name} | MODEL: {card_model_name}",
            text_color="#3B8ED0", font=("Arial", 12, "bold")
        ).pack(side="left", padx=20)
        ctk.CTkLabel(header, text="■ Nekrotik Çekirdek",
                     text_color="#FF4D4D", font=("Arial", 12, "bold")).pack(side="left", padx=8)
        ctk.CTkLabel(header, text="■ Peritümoral Ödem",
                     text_color="#4BB543", font=("Arial", 12, "bold")).pack(side="left", padx=8)
        ctk.CTkLabel(header, text="■ Aktif Tümör",
                     text_color="#57bbff", font=("Arial", 12, "bold")).pack(side="left", padx=8)

        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=10)

        # Metrik paneli
        metrics_panel = ctk.CTkFrame(
            content, fg_color="#1A1A1B", width=300,
            corner_radius=10, border_width=1, border_color="#444"
        )
        metrics_panel.pack(side="right", padx=10, pady=10, fill="y")
        ctk.CTkLabel(
            metrics_panel, text="KATMAN METRİKLERİ",
            font=("Arial", 11, "bold"), text_color="gray"
        ).pack(pady=(10, 5))

        lbl_nekrotik = ctk.CTkLabel(metrics_panel, text="", text_color="#FF4D4D",
                                    font=("Consolas", 11), justify="left")
        lbl_nekrotik.pack(padx=10, pady=4, anchor="w")
        lbl_odem = ctk.CTkLabel(metrics_panel, text="", text_color="#4BB543",
                                font=("Consolas", 11), justify="left")
        lbl_odem.pack(padx=10, pady=4, anchor="w")
        lbl_aktif = ctk.CTkLabel(metrics_panel, text="", text_color="#57bbff",
                                 font=("Consolas", 11), justify="left")
        lbl_aktif.pack(padx=10, pady=4, anchor="w")

        # Görseller
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='#161617')
        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.get_tk_widget().pack(side="left", expand=True)

        layer_label = ctk.CTkLabel(card, text="", font=("Arial", 12, "italic"))
        layer_label.pack()

        # ── Metrik hesaplama ──────────────────────────────────────────────
        def calc_metrics(gt_bin, pr_bin):
            inter = np.sum(gt_bin * pr_bin)
            s_gt  = np.sum(gt_bin)
            s_pr  = np.sum(pr_bin)
            dice  = (2. * inter) / (s_gt + s_pr + 1e-7)
            acc   = np.mean(gt_bin == pr_bin)
            prec  = inter / (s_pr + 1e-7)
            rec   = inter / (s_gt + 1e-7)
            return (f"  DICE : {dice:.3f}\n"
                    f"  ACC  : {acc:.3f}\n"
                    f"  PREC : {prec:.3f}\n"
                    f"  REC  : {rec:.3f}")

        # ── Slider güncelleme ─────────────────────────────────────────────
        def update(val):
            idx = int(float(val))
            layer_label.configure(text=f"Katman: {idx + 1} / {depth}")
            pr = preds[idx]

            if local_gt is not None:
                gt = local_gt[:, :, idx]
                lbl_nekrotik.configure(
                    text=f"NEKROTİK ÇEKİRDEK (sınıf 1):\n"
                         + calc_metrics((gt == 1).astype(np.uint8), (pr == 1).astype(np.uint8))
                )
                lbl_odem.configure(
                    text=f"PERİTÜMORAL ÖDEM (sınıf 2):\n"
                         + calc_metrics((gt == 2).astype(np.uint8), (pr == 2).astype(np.uint8))
                )
                lbl_aktif.configure(
                    text=f"AKTİF TÜMÖR (sınıf 3):\n"
                         + calc_metrics((gt == 3).astype(np.uint8), (pr == 3).astype(np.uint8))
                )
            else:
                lbl_nekrotik.configure(text="Maske yüklenmedi")
                lbl_odem.configure(text="")
                lbl_aktif.configure(text="")

            for ax in [ax1, ax2]:
                ax.clear()
                ax.axis('off')

            img = local_flair[:, :, idx].T

            ax1.set_title(
                "Ground Truth" if local_gt is not None else "FLAIR",
                color='white', fontsize=9
            )
            ax1.imshow(img, cmap='gray', origin='lower')
            if local_gt is not None:
                gt_s = local_gt[:, :, idx].T
                ax1.imshow(
                    np.ma.masked_where(gt_s == 0, gt_s),
                    cmap=self.multi_color_cmap, alpha=0.6,
                    origin='lower', vmin=0, vmax=3
                )

            
            ax2.set_title(
                f"Tahmin — {card_model_name}",
                color='white', fontsize=9
            )
            ax2.imshow(img, cmap='gray', origin='lower')
            pr_show = pr.T
            ax2.imshow(
                np.ma.masked_where(pr_show == 0, pr_show),
                cmap=self.multi_color_cmap, alpha=0.6,
                origin='lower', vmin=0, vmax=3
            )

            canvas.draw_idle()

        slider = ctk.CTkSlider(card, from_=0, to=depth - 1, command=update, width=500)
        slider.pack(pady=15)
        slider.set(depth // 2)
        update(depth // 2)


if __name__ == "__main__":
    app = NeuroScanApp()
    app.mainloop()