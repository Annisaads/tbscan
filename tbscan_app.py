import streamlit as st
import os
import numpy as np  
from model_handler import ModelHandler
from prediction_handler import PredictionHandler
from evaluation_handler import EvaluationHandler

class UIHandler:
    """Kelas untuk menangani tampilan UI aplikasi."""
    
    def __init__(self):
        """Inisialisasi UIHandler."""
        self.css_loaded = False
        self.header_displayed = False
        self.load_css()
    
    def load_css(self):
        """Menambahkan CSS lokal."""
        if not self.css_loaded:
            if os.path.exists("style.css"):
                with open("style.css") as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                self.css_loaded = True
    
    def display_header(self):
        """Menampilkan header aplikasi."""
        if not self.header_displayed:
            st.markdown("<h1 style='text-align: center; color: #001B48;'>🩺 TBScan</h1>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>Aplikasi ini dirancang untuk memprediksi citra Chest X-Ray yang menunjukkan kondisi normal atau tuberkulosis, serta menampilkan evaluasi kinerja model klasifikasi tuberkulosis berdasarkan citra Chest X-Ray yang telah melalui proses pelatihan sebelumnya.</div>", unsafe_allow_html=True)
            st.markdown("---")
            self.header_displayed = True
    
    def display_navigation_buttons(self):
        """Menampilkan tombol navigasi."""
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Prediksi", use_container_width=True):
                st.session_state["page"] = "prediksi"
        with col2:
            if st.button("📈 Evaluasi", use_container_width=True):
                st.session_state["page"] = "evaluasi"
    
    def display_prediction_result(self, predicted_class, confidence):
        """Tampilkan hasil prediksi beserta nilai confidence.""" 
        # Convert NumPy value to Python float before rounding
        try:
            # Ensure confidence is a Python float
            import numpy as np
            if isinstance(confidence, np.ndarray):
                confidence = float(confidence.item() if confidence.size == 1 else confidence[0])
            else:
                confidence = float(confidence)
                
            confidence_percent = round(confidence * 100, 2)
            
            if predicted_class == 1:
                result_html = self.generate_prediction_html(
                    "Tuberkulosis", 
                    "🚨 Tuberkulosis", 
                    f"Hasil prediksi menunjukkan kemungkinan adanya infeksi tuberkulosis pada paru-paru "
                    f"dengan tingkat keyakinan sebesar {confidence_percent}%. "
                    "Sebaiknya segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut."
                )
            else:
                result_html = self.generate_prediction_html(
                    "Normal", 
                    "✅ Normal", 
                    f"Hasil prediksi menunjukkan bahwa paru-paru dalam kondisi sehat dan bebas dari tuberkulosis "
                    f"dengan tingkat keyakinan sebesar {confidence_percent}%. "
                    "Selalu perhatikan kesehatan Anda dan lakukan pemeriksaan secara rutin untuk menjaga kondisi paru-paru tetap baik."
                )

            st.markdown(result_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat menampilkan hasil prediksi: {e}")
            st.warning(f"Predicted class: {predicted_class}, Confidence type: {type(confidence)}")

    def generate_prediction_html(self, status, title, message):
        """Buat HTML untuk hasil prediksi."""
        class_name = "prediction-tuberculosis" if status == "Tuberkulosis" else "prediction-normal"
        return f"""
            <div class="prediction-box {class_name}">
                {title}
                <div class="prediction-message prediction-{status.lower()}-message">
                    {message}
                </div>
            </div>
        """
        
    def display_model_info(self, architecture, learning_rate, batch_size):
        """Menampilkan informasi model yang digunakan."""
        if architecture and learning_rate and batch_size:
            st.markdown("---")
            st.markdown("### ⚙️ Konfigurasi Aktif")
            st.info(f"Arsitektur: `{architecture}` | Batch Size: `{batch_size}` | Learning Rate: `{learning_rate}`")
            st.markdown("---")


# Konfigurasi halaman
st.set_page_config(page_title="TBScan", page_icon="🩺")

def main():
    """Fungsi utama aplikasi."""
    # Inisialisasi handler
    ui = UIHandler()
    model = ModelHandler()
    predictor = PredictionHandler(ui)
    evaluator = EvaluationHandler(ui)
    
    # Tampilkan header aplikasi
    ui.display_header()
    
    # Tampilkan dan atur konfigurasi di sidebar
    model_config = model.sidebar_konfigurasi()
    
    # Tampilkan tombol navigasi
    ui.display_navigation_buttons()
    
    # Tampilkan halaman sesuai pilihan
    if "page" not in st.session_state:
        st.session_state["page"] = "prediksi"
        
    if st.session_state["page"] == "prediksi":
        predictor.prediction_page(model_config)
    elif st.session_state["page"] == "evaluasi":
        evaluator.evaluation_page(model_config)

if __name__ == "__main__":
    main()