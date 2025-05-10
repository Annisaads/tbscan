import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from model_handler import ModelHandler

class EvaluationHandler:
    """Kelas untuk menangani fitur evaluasi."""
    
    def __init__(self, ui_handler):
        """Inisialisasi EvaluationHandler."""
        self.ui = ui_handler
        self.model_handler = ModelHandler()
        
    def evaluation_page(self, model_config):
        """Fungsi untuk halaman evaluasi model sesuai sequence diagram."""
        self.architecture, self.learning_rate, self.batch_size = model_config

        st.title("📈 Evaluasi Kinerja Model")
        st.write("Evaluasi ini akan menampilkan hasil kinerja model klasifikasi tuberkulosis pada citra Chest X-Ray yang telah melalui proses pelatihan sebelumnya.")

        # Tampilkan konfigurasi model jika sudah disimpan
        if st.session_state.get("config_saved"):
            st.info(f"**Konfigurasi Aktif:**  \nArsitektur: `{self.architecture}` | Batch Size: `{self.batch_size}` | Learning Rate: `{self.learning_rate}`", icon="⚙️")
        else:
            # Sesuai sequence diagram alt flow: konfigurasi model tidak lengkap
            st.warning("⚠️ Pilih konfigurasi model terlebih dahulu!")
            return

        # Path file history npz
        history_path = f"./model6/history_{self.architecture}_lr{self.learning_rate}_bs{self.batch_size}.npz"

        if not os.path.exists(history_path):
            st.warning("⚠️ File evaluasi model tidak ditemukan!")
            return

        # Tombol Evaluasi sesuai sequence diagram
        if st.button("🔎 Evaluasi"):
            # Sesuai sequence diagram: Memuat data evaluasi
            evaluation_data = self.load_evaluation_data(history_path)
            if evaluation_data:
                # Evaluasi model sesuai sequence diagram
                self.evaluate_model(evaluation_data)
                
    def load_evaluation_data(self, history_path):
        """Memuat data evaluasi dari file npz - sesuai sequence diagram."""
        try:
            # Load file npz
            data = np.load(history_path, allow_pickle=True)
            return data
        except Exception as e:
            st.error(f"❌ Gagal memuat data evaluasi: {e}")
            return None
    
    def evaluate_model(self, data):
        """Evaluasi model menggunakan data yang telah dimuat - sesuai sequence diagram."""
        try:
            # Ambil data dari file npz
            y_true = data['y_true']
            y_pred = data['y_pred']
            class_names = data['class_names'].tolist()

            # Menghitung metrik evaluasi
            evaluation_results = {
                "Accuracy": [accuracy_score(y_true, y_pred)],
                "Precision": [precision_score(y_true, y_pred)],
                "Recall": [recall_score(y_true, y_pred)],
                "F1-Score": [f1_score(y_true, y_pred)]
            }

            # Menampilkan hasil evaluasi - sesuai sequence diagram
            self.display_evaluation_results(evaluation_results, class_names, y_true, y_pred)

            # Tampilkan grafik history - sesuai sequence diagram
            self.display_training_history(data)
        except Exception as e:
            st.error(f"❌ Gagal mengevaluasi model: {e}")

    def display_evaluation_results(self, evaluation_results, class_names, y_true, y_pred):
        """Tampilkan hasil evaluasi model - sesuai sequence diagram."""
        # Tampilkan Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Menampilkan hasil metrik evaluasi
        st.subheader("📋 Laporan Evaluasi")
        for metric, value in evaluation_results.items():
            st.write(f"**{metric}:** {value[0] * 100:.2f}%")

    def display_training_history(self, data):
        """Tampilkan grafik akurasi dan loss dari file npz - sesuai sequence diagram."""
        st.subheader("📈 Grafik Akurasi dan Loss Selama Pelatihan")

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        ax[0].plot(data['history'].item().get('accuracy'), label='Training Accuracy')
        if 'val_accuracy' in data['history'].item():
            ax[0].plot(data['history'].item()['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()
        ax[0].grid(True)

        # Loss
        ax[1].plot(data['history'].item().get('loss'), label='Training Loss')
        if 'val_loss' in data['history'].item():
            ax[1].plot(data['history'].item()['val_loss'], label='Validation Loss')
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        ax[1].grid(True)

        st.pyplot(fig)

