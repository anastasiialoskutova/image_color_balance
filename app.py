import sys
import os
import json
import cv2
import numpy as np
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QMessageBox, QScrollArea, QGroupBox, QFormLayout,
    QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# -----------------------
# Алгоритми корекції
# -----------------------

def gray_world(img):
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = [np.mean(img_float[:,:,i]) for i in range(3)]
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    kb = avg_gray / (avg_b + 1e-8)
    kg = avg_gray / (avg_g + 1e-8)
    kr = avg_gray / (avg_r + 1e-8)
    out = img_float.copy()
    out[:,:,0] *= kb
    out[:,:,1] *= kg
    out[:,:,2] *= kr
    return np.clip(out, 0, 255).astype(np.uint8)

def max_rgb_white_patch(img):
    img_float = img.astype(np.float32)
    max_b = np.max(img_float[:,:,0])
    max_g = np.max(img_float[:,:,1])
    max_r = np.max(img_float[:,:,2])
    avg_max = (max_b + max_g + max_r) / 3.0 + 1e-8
    kb = avg_max / (max_b + 1e-8)
    kg = avg_max / (max_g + 1e-8)
    kr = avg_max / (max_r + 1e-8)
    out = img_float.copy()
    out[:,:,0] *= kb
    out[:,:,1] *= kg
    out[:,:,2] *= kr
    return np.clip(out, 0, 255).astype(np.uint8)

def shades_of_gray(img, p=6):
    img_float = img.astype(np.float32)
    b = np.power(np.mean(np.power(img_float[:,:,0], p)), 1.0/p)
    g = np.power(np.mean(np.power(img_float[:,:,1], p)), 1.0/p)
    r = np.power(np.mean(np.power(img_float[:,:,2], p)), 1.0/p)
    avg = (b + g + r) / 3.0 + 1e-8
    kb = avg / (b + 1e-8)
    kg = avg / (g + 1e-8)
    kr = avg / (r + 1e-8)
    out = img_float.copy()
    out[:,:,0] *= kb
    out[:,:,1] *= kg
    out[:,:,2] *= kr
    return np.clip(out, 0, 255).astype(np.uint8)

def lab_equalize(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return out

def hist_eq(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return out

def single_scale_retinex(img, sigma=30):
    img_float = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_float, (0,0), sigma)
    retinex = np.log(img_float) - np.log(blur + 1e-8)
    for i in range(3):
        r = retinex[:,:,i]
        r = (r - r.min()) / (r.max() - r.min() + 1e-8) * 255.0
        retinex[:,:,i] = r
    return np.clip(retinex, 0, 255).astype(np.uint8)

# -----------------------
# Утиліти для корекцій
# -----------------------

def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    img_float = img.astype(np.float32)
    out = img_float * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)

def adjust_saturation(img, sat_factor=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= sat_factor
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def adjust_temperature(img, temp_shift=0):
    out = img.astype(np.float32)
    out[:,:,2] = out[:,:,2] + temp_shift  # R
    out[:,:,0] = out[:,:,0] - temp_shift  # B
    return np.clip(out, 0, 255).astype(np.uint8)

# -----------------------
# Головний клас програми
# -----------------------

class ColorBalanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Додаток для виправлення кольорового дисбалансу зображень")
        self.resize(1200, 760)
        self.setAcceptDrops(True)

        self.image_paths = []
        self.current_index = 0
        self.original_img = None  # BGR uint8
        self.corrected_img = None

        self.cache_algo = {}

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # --- Прев'ю ---
        preview_layout = QHBoxLayout()
        self.label_original = QLabel("Оригінал")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setFixedSize(520, 520)
        self.label_original.setStyleSheet("border:1px solid #aaa; background:#fff;")

        self.label_corrected = QLabel("Виправлено")
        self.label_corrected.setAlignment(Qt.AlignCenter)
        self.label_corrected.setFixedSize(520, 520)
        self.label_corrected.setStyleSheet("border:1px solid #aaa; background:#fff;")

        preview_layout.addWidget(self.label_original)
        preview_layout.addWidget(self.label_corrected)

        # Split slider
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setMinimum(0)
        self.split_slider.setMaximum(100)
        self.split_slider.setValue(100)
        self.split_slider.setToolTip("0 = повністю оригінал, 100 = повністю скориговано")
        self.split_slider.valueChanged.connect(self.update_split_preview)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_font = QFont("Arial", 11)

        self.btn_open = QPushButton("Відкрити фото")
        self.btn_open.setFont(btn_font)
        self.btn_open.clicked.connect(self.open_images)

        self.btn_save = QPushButton("Зберегти результат")
        self.btn_save.setFont(btn_font)
        self.btn_save.clicked.connect(self.save_result)

        self.btn_batch = QPushButton("Пакетна обробка (папка)")
        self.btn_batch.setFont(btn_font)
        self.btn_batch.clicked.connect(self.batch_process_folder)

        self.btn_save_preset = QPushButton("Зберегти пресет")
        self.btn_save_preset.clicked.connect(self.save_preset)

        self.btn_load_preset = QPushButton("Завантажити пресет")
        self.btn_load_preset.clicked.connect(self.load_preset)

        for b in (self.btn_open, self.btn_save, self.btn_batch, self.btn_save_preset, self.btn_load_preset):
            b.setFont(btn_font)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border-radius: 6px;
                    padding: 8px 14px;
                }
                QPushButton:hover { background-color: #2980b9; }
            """)
            btn_layout.addWidget(b)

        left_col.addLayout(preview_layout)
        left_col.addWidget(QLabel("Позиція Before/After (0 = оригінал, 100 = скориговано)"))
        left_col.addWidget(self.split_slider)
        left_col.addLayout(btn_layout)

        # -----------------------------
        # Праворуч: слайдери алгоритмів та корекцій
        # -----------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        scroll.setWidget(right_widget)

        alg_group = QGroupBox("Зважування алгоритмів (змішування)")
        alg_form = QFormLayout()

        self.alg_sliders = {}
        algs = [
            ("Gray World", "gray"),
            ("Max RGB (White Patch)", "maxrgb"),
            ("Shades of Gray (p=6)", "shades"),
            ("LAB Equalize", "lab"),
            ("Histogram EQ", "hist"),
            ("Single-Scale Retinex", "retinex")
        ]
        for name, key in algs:
            s = QSlider(Qt.Horizontal)
            s.setMinimum(0); s.setMaximum(100); s.setValue(0)
            s.valueChanged.connect(self.on_algo_slider_changed)
            alg_form.addRow(QLabel(name), s)
            self.alg_sliders[key] = s

        alg_group.setLayout(alg_form)
        right_layout.addWidget(alg_group)

        manual_group = QGroupBox("Ручні корекції")
        manual_form = QFormLayout()

        self.sld_brightness = QSlider(Qt.Horizontal)
        self.sld_brightness.setMinimum(-100); self.sld_brightness.setMaximum(100); self.sld_brightness.setValue(0)
        self.sld_brightness.valueChanged.connect(self.update_all)
        manual_form.addRow(QLabel("Яскравість (-100..100)"), self.sld_brightness)

        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setMinimum(0); self.sld_contrast.setMaximum(300); self.sld_contrast.setValue(100)
        self.sld_contrast.valueChanged.connect(self.update_all)
        manual_form.addRow(QLabel("Контраст (0..300%)"), self.sld_contrast)

        self.sld_saturation = QSlider(Qt.Horizontal)
        self.sld_saturation.setMinimum(0); self.sld_saturation.setMaximum(300); self.sld_saturation.setValue(100)
        self.sld_saturation.valueChanged.connect(self.update_all)
        manual_form.addRow(QLabel("Насиченість (0..300%)"), self.sld_saturation)

        self.sld_temp = QSlider(Qt.Horizontal)
        self.sld_temp.setMinimum(-100); self.sld_temp.setMaximum(100); self.sld_temp.setValue(0)
        self.sld_temp.valueChanged.connect(self.update_all)
        manual_form.addRow(QLabel("Температура (-100..100)"), self.sld_temp)

        manual_group.setLayout(manual_form)
        right_layout.addWidget(manual_group)

        preset_name_layout = QHBoxLayout()
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Ім'я пресету (наприклад: 'Sunny')")
        preset_name_layout.addWidget(self.preset_name_edit)
        right_layout.addLayout(preset_name_layout)

        help_lbl = QLabel("Drag & Drop: перетягни файли у вікно.\n"
                          "Split slider дозволяє порівняти оригінал і скориговане.\n"
                          "Налаштуй ваги алгоритмів і ручні корекції, потім - пакетна обробка.")
        help_lbl.setWordWrap(True)
        right_layout.addWidget(help_lbl)
        right_layout.addStretch(1)

        right_col.addWidget(scroll)
        main_layout.addLayout(left_col, 2)
        main_layout.addLayout(right_col, 1)
        self.setLayout(main_layout)

        # початкові зв'язки
        self.split_slider.setValue(100)
        self.update_all()

    # -----------------------
    # Завантаження / Drag&Drop
    # -----------------------
    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Виберіть фото", "", "Images (*.jpg *.png *.jpeg *.bmp *.tiff)")
        if paths:
            self.image_paths = paths
            self.current_index = 0
            self.load_current_image()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        added = False
        for u in urls:
            p = u.toLocalFile()
            if os.path.isfile(p) and p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff')):
                self.image_paths.append(p)
                added = True
        if added:
            self.current_index = 0
            self.load_current_image()

    def load_current_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Помилка", f"Не вдалося відкрити {path}")
            return
        self.original_img = img
        self.cache_algo = {}
        self.show_image_in_label(self.original_img, self.label_original)
        # IMPORTANT: do not overwrite label_corrected here; let split_preview decide what to show
        self.update_all()

    # -----------------------
    # Алгоритми / Змішування
    # -----------------------
    def compute_algo(self, key):
        if self.original_img is None:
            return None
        if key in self.cache_algo:
            return self.cache_algo[key]
        if key == "gray":
            out = gray_world(self.original_img)
        elif key == "maxrgb":
            out = max_rgb_white_patch(self.original_img)
        elif key == "shades":
            out = shades_of_gray(self.original_img, p=6)
        elif key == "lab":
            out = lab_equalize(self.original_img)
        elif key == "hist":
            out = hist_eq(self.original_img)
        elif key == "retinex":
            out = single_scale_retinex(self.original_img, sigma=30)
        else:
            out = self.original_img.copy()
        self.cache_algo[key] = out
        return out

    def on_algo_slider_changed(self, _=None):
        self.update_all()

    def update_all(self):
        if self.original_img is None:
            return

        # Отримуємо ваги алгоритмів (0..1)
        weights = {}
        total = 0.0
        for k, s in self.alg_sliders.items():
            w = s.value() / 100.0
            weights[k] = w
            total += w

        if total > 1.0 and total > 0:
            for k in weights:
                weights[k] /= total
            total = 1.0

        orig_w = max(0.0, 1.0 - sum(weights.values()))

        h_img = self.original_img.astype(np.float32) * orig_w
        for k, w in weights.items():
            if w <= 0:
                continue
            alg_img = self.compute_algo(k).astype(np.float32)
            h_img += alg_img * w

        blended = np.clip(h_img, 0, 255).astype(np.uint8)

        # Ручні корекції — ПІДСИЛЕНІ для помітного ефекту
        brightness = self.sld_brightness.value()  # -100..100 (add)
        # contrast slider 0..300 -> map to multiplier 0..3, but stronger mapping used:
        contrast = 1 + (self.sld_contrast.value() - 100) / 100.0 * 2.0
        # saturation 0..300 -> map to strong factor: 1 + (v-100)/100*3
        sat = 1 + (self.sld_saturation.value() - 100) / 100.0 * 3.0
        # temperature -100..100 -> scale to -60..60
        temp = int(self.sld_temp.value() * 0.6)

        # contrast multiplicative (centered at 128), brightness additive
        blended = np.clip((blended.astype(np.float32) - 128.0) * contrast + 128.0 + brightness, 0, 255).astype(np.uint8)
        blended = adjust_saturation(blended, sat_factor=sat)
        blended = adjust_temperature(blended, temp_shift=temp)

        self.corrected_img = blended
        # IMPORTANT: do not call show_image_in_label(corrected) here to avoid overwriting split preview
        self.update_split_preview()

    # -----------------------
    # Before/After split preview
    # -----------------------
    def update_split_preview(self):
        if self.original_img is None:
            return

        left = self.original_img
        right = self.corrected_img if self.corrected_img is not None else self.original_img

        split_pct = self.split_slider.value() / 100.0  # 0..1

        target_w = self.label_corrected.width()
        target_h = self.label_corrected.height()

        ori_scaled = self.scale_to_size(left, target_w, target_h)
        corr_scaled = self.scale_to_size(right, target_w, target_h)

        w = ori_scaled.shape[1]
        split_x = int(w * split_pct)

        # Edge cases: fully original or fully corrected
        if split_pct <= 0.001:
            self.show_image_in_label(ori_scaled, self.label_corrected)
            return
        if split_pct >= 0.999:
            self.show_image_in_label(corr_scaled, self.label_corrected)
            return

        combined = ori_scaled.copy()
        combined[:, split_x:] = corr_scaled[:, split_x:]
        self.show_image_in_label(combined, self.label_corrected)

    # -----------------------
    # Відображення в QLabel
    # -----------------------
    def scale_to_size(self, img, max_w, max_h):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        x = (max_w - new_w) // 2
        y = (max_h - new_h) // 2
        canvas[y:y+new_h, x:x+new_w] = scaled
        return canvas

    def show_image_in_label(self, img, label):
        qimg = self.cv2_to_qimage(img)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def cv2_to_qimage(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg

    # -----------------------
    # Збереження фінального зображення
    # -----------------------
    def save_result(self):
        if self.corrected_img is None:
            QMessageBox.information(self, "Немає зображення", "Спочатку відкрийте зображення та налаштуйте параметри.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Виберіть папку для збереження")
        if not folder:
            return
        src_path = self.image_paths[self.current_index] if self.image_paths else "image"
        base, ext = os.path.splitext(os.path.basename(src_path))
        save_path = os.path.join(folder, f"{base}_corrected{ext if ext else '.png'}")
        cv2.imwrite(save_path, self.corrected_img)
        QMessageBox.information(self, "Збережено", f"Збережено: {save_path}")

    # -----------------------
    # Пакетна обробка
    # -----------------------
    def batch_process_folder(self):
        if not self.image_paths:
            QMessageBox.information(self, "Немає зображень", "Спочатку додайте або відкрийте зображення або папку.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Виберіть папку для збереження результатів")
        if not folder:
            return
        count = 0
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            self.original_img = img
            self.cache_algo = {}
            self.update_all()
            base, ext = os.path.splitext(os.path.basename(path))
            save_path = os.path.join(folder, f"{base}_batch_corrected{ext if ext else '.png'}")
            cv2.imwrite(save_path, self.corrected_img)
            count += 1
        QMessageBox.information(self, "Готово", f"Пакетна обробка завершена. Оброблено: {count} файлів.\nЗбережено в {folder}")

    # -----------------------
    # Пресети
    # -----------------------
    def gather_current_params(self):
        params = {}
        params['algs'] = {k: s.value() for k, s in self.alg_sliders.items()}
        params['brightness'] = self.sld_brightness.value()
        params['contrast'] = self.sld_contrast.value()
        params['saturation'] = self.sld_saturation.value()
        params['temperature'] = self.sld_temp.value()
        return params

    def apply_params(self, params):
        for k, v in params.get('algs', {}).items():
            if k in self.alg_sliders:
                self.alg_sliders[k].setValue(int(v))
        self.sld_brightness.setValue(int(params.get('brightness', 0)))
        self.sld_contrast.setValue(int(params.get('contrast', 100)))
        self.sld_saturation.setValue(int(params.get('saturation', 100)))
        self.sld_temp.setValue(int(params.get('temperature', 0)))
        self.cache_algo = {}
        self.update_all()

    def save_preset(self):
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Ім'я пресету", "Введіть ім'я пресету у полі зверху.")
            return
        params = self.gather_current_params()
        presets_dir = os.path.join(os.getcwd(), "presets")
        os.makedirs(presets_dir, exist_ok=True)
        fname = os.path.join(presets_dir, f"{name}.json")
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Збережено", f"Пресет збережено у {fname}")

    def load_preset(self):
        presets_dir = os.path.join(os.getcwd(), "presets")
        os.makedirs(presets_dir, exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(self, "Завантажити пресет", presets_dir, "JSON files (*.json)")
        if not fname:
            return
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.apply_params(params)
            QMessageBox.information(self, "Завантажено", f"Пресет завантажено: {os.path.basename(fname)}")
        except Exception as e:
            QMessageBox.warning(self, "Помилка", f"Не вдалося завантажити пресет:\n{e}")

# -----------------------
# Запуск програми
# -----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorBalanceApp()
    window.show()
    sys.exit(app.exec_())
