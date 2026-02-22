import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QComboBox, QMessageBox, QFrame
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QLabel
from PyQt5.QtWidgets import QProgressBar



class GW_Predictor_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊  پیش‌بینی تراز آب زیرزمینی با استفاده از هوش مصنوعی")
        self.setGeometry(400, 200, 600, 500)

        # مسیر برای PyInstaller
        BASEDIR = getattr(sys, '_MEIPASS', os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        bg_img_path = os.path.join(BASEDIR, "Picture1.png")

        # ----- بنر بالای برنامه (همان تصویر پس‌زمینه کوچک‌شده) -----
        self.banner = QLabel(self)
        self.banner.setPixmap(QPixmap(bg_img_path).scaled(520, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("background: rgba(255,255,255,180); border-radius: 12px; margin-bottom: 12px;")

        # ----- چیدمان اصلی -----
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(28, 28, 28, 28)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # حالت نامعین (infinite loading)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)  # اول مخفی باشه
        
        main_layout.addWidget(self.banner, alignment=Qt.AlignCenter)

        # ----- باکس نیمه شفاف برای ابزارها -----
        tool_widget = QWidget(self)
        tool_widget.setStyleSheet("background: rgba(255,255,255,200); border-radius: 16px;")
        layout = QVBoxLayout(tool_widget)
                # نوار پیشرفت
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # حالت indeterminate
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.setSpacing(15)
        layout.setContentsMargins(20, 18, 20, 18)

        font_label = QFont("B Yekan", 12)
        font_button = QFont("B Yekan", 10)

        # مدل
        self.model_label = QLabel("🔹 انتخاب مدل:", self)
        self.model_label.setFont(font_label)
        self.model_combo = QComboBox(self)
        self.model_combo.setFont(font_button)
        self.model_combo.addItems(["CNN", "LSTM"])
        self.model_combo.setStyleSheet("padding: 7px; background:#e7f5ff; border-radius:8px;")

        # سناریو
        self.shift_label = QLabel("🔹 انتخاب سناریو:", self)
        self.shift_label.setFont(font_label)
        self.shift_combo = QComboBox(self)
        self.shift_combo.setFont(font_button)
        self.shift_combo.addItems(["GWL", "GWLt-1"])
        self.shift_combo.setStyleSheet("padding: 7px; background:#e7f5ff; border-radius:8px;")

        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.shift_label)
        layout.addWidget(self.shift_combo)

        layout.addWidget(self._line())
        # Learning rate - مقدار ثابت
        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("Learning rate (مقداری بین 0ـ1 (مثلاً 0.001))")
        layout.addWidget(self.lr_input)

        # سایر پارامترها به صورت بازه
        self.seq_min, self.seq_max = self.add_param_range_field("Sequence length:", layout)
        self.dense_min, self.dense_max = self.add_param_range_field("Dense size:", layout)
        self.batch_min, self.batch_max = self.add_param_range_field("Batch size:", layout)
        self.filters_min, self.filters_max = self.add_param_range_field("Filters:", layout)

        
        # دکمه‌ها
        self.btn_gw = QPushButton("📂 انتخاب فایل تراز آب ", self)
        self.btn_gw.setFont(font_button)
        self.btn_gw.clicked.connect(self.load_gw_file)
        self.btn_gw.setStyleSheet("background:#f6e58d; border-radius:7px; padding:8px;")

        self.btn_weather = QPushButton("📂 انتخاب فایل داده‌های هواشناسی", self)
        self.btn_weather.setFont(font_button)
        self.btn_weather.clicked.connect(self.load_weather_file)
        self.btn_weather.setStyleSheet("background:#f6e58d; border-radius:7px; padding:8px;")

        self.btn_output = QPushButton("📁 انتخاب پوشه ذخیره نتایج", self)
        self.btn_output.setFont(font_button)
        self.btn_output.clicked.connect(self.select_output_dir)
        self.btn_output.setStyleSheet("background:#dff9fb; border-radius:7px; padding:8px;")

        layout.addWidget(self.btn_gw)
        layout.addWidget(self.btn_weather)
        layout.addWidget(self.btn_output)

        layout.addWidget(self._line())

        # دکمه اجرا
        self.run_btn = QPushButton("🚀 اجرای پیش‌بینی", self)
        self.run_btn.setFont(QFont("B yekan", 16, QFont.Bold))
        self.run_btn.clicked.connect(self.run_model)
        self.run_btn.setStyleSheet("background-color: #22a7f0; color: white; padding: 12px; border-radius: 13px; font-weight:bold;")
        layout.addWidget(self.run_btn, alignment=Qt.AlignCenter)

        main_layout.addWidget(tool_widget)
        self.setLayout(main_layout)

        self.gw_path = ""
        self.weather_path = ""
        self.output_dir = ""

    def _line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        return line
    def add_param_range_field(self, label_text, layout):
        label = QLabel(label_text)
        label.setFont(QFont("B Yekan", 10))
        hbox = QHBoxLayout()
        min_input = QLineEdit()
        min_input.setPlaceholderText("min")
        max_input = QLineEdit()
        max_input.setPlaceholderText("max")
        min_input.setStyleSheet("padding: 5px;")
        max_input.setStyleSheet("padding: 5px;")
        hbox.addWidget(label)
        hbox.addWidget(min_input)
        hbox.addWidget(max_input)
        layout.addLayout(hbox)
        return min_input, max_input
    
    def resize_bg(self, event):
        self.bg_label.setPixmap(QPixmap(self.bg_img_path).scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.banner.setPixmap(QPixmap(self.bg_img_path).scaled(self.banner.width(), 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_gw_file(self):
        self.gw_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل تراز آب", "", "CSV files (*.csv)")
        if self.gw_path:
            QMessageBox.information(self, "✅ فایل تراز آب انتخاب شد", f"{self.gw_path}")

    def load_weather_file(self):
        self.weather_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل هواشناسی", "", "CSV files (*.csv)")
        if self.weather_path:
            QMessageBox.information(self, "✅ فایل هواشناسی انتخاب شد", f"{self.weather_path}")

    def select_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "انتخاب پوشه خروجی")
        if self.output_dir:
            QMessageBox.information(self, "✅ پوشه خروجی انتخاب شد", f"{self.output_dir}")

    def run_model(self):
    # خواندن learning_rate
        lr = self.lr_input.text()

        # خواندن بازه‌ها
        seq_min = self.seq_min.text()
        seq_max = self.seq_max.text()
        dense_min = self.dense_min.text()
        dense_max = self.dense_max.text()
        batch_min = self.batch_min.text()
        batch_max = self.batch_max.text()
        filters_min = self.filters_min.text()
        filters_max = self.filters_max.text()

        # چک کردن ورودها
        if not all([self.gw_path, self.weather_path, self.output_dir]):
            QMessageBox.warning(self, "⚠️ خطا", "لطفاً مسیر همه فایل‌ها را مشخص کنید.")
            return

        if not all([lr, seq_min, seq_max, dense_min, dense_max, batch_min, batch_max, filters_min, filters_max]):
            QMessageBox.warning(self, "⚠️ خطا", "همه فیلدهای هایپرپارامتر باید تکمیل شوند.")
            return

        model = self.model_combo.currentText()
        shift = self.shift_combo.currentText()
        model_script = f"{model}_seq2val{'_GWLshift' if shift == 'GWLt-1' else ''}.py"

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(base_dir, "models", model_script)

        # 👇 شروع نمایش نوار پیشرفت
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)  # حالت نامشخص (چرخش بی‌نهایت)

        try:
            subprocess.run([
                "python", model_path,
                self.gw_path,
                self.weather_path,
                self.output_dir,
                lr,
                seq_min, seq_max,
                dense_min, dense_max,
                batch_min, batch_max,
                filters_min, filters_max
            ], check=True)

            QMessageBox.information(self, "✅ موفقیت", "مدل اجرا و نتایج ذخیره شدند!")

        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "❌ خطا", f"{e}")

        finally:
            # 👇 پایان نوار پیشرفت
            self.progress_bar.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GW_Predictor_GUI()
    win.show()
    sys.exit(app.exec_())
