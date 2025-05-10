import streamlit as st
import os
import tensorflow as tf
from model_handler import ModelHandler

class PredictionHandler:
    """Kelas untuk menangani fitur prediksi."""
    
    def __init__(self, ui_handler):
        """Inisialisasi PredictionHandler."""
        self.ui = ui_handler
        self.model_handler = ModelHandler()
    
    def prediction_page(self, model_config):
        """Fungsi untuk menampilkan halaman prediksi sesuai sequence diagram."""
        architecture, learning_rate, batch_size = model_config
        
        st.title("🔍 Prediksi Tuberkulosis pada Citra Chest X-ray")
        st.header("📂 Unggah Gambar X-ray")
        
        # Tampilkan halaman prediksi - sesuai sequence diagram
        uploaded_file = st.file_uploader("Pilih gambar untuk diklasifikasikan", type=["jpg", "png", "jpeg"])
        
        if not uploaded_file:
            return
            
        # Validasi gambar yang diunggah - sesuai sequence diagram alt flow
        # Tampilkan gambar jika valid
        st.image(uploaded_file, caption="🖼️ Gambar yang diunggah", use_container_width=True)
        
        # Pilih konfigurasi model - sesuai sequence diagram
        if not (architecture and learning_rate and batch_size):
            st.warning("⚠️ Harap lengkapi konfigurasi model terlebih dahulu di sidebar.")
            return

        # Tampilkan informasi model - sesuai sequence diagram
        self.ui.display_model_info(architecture, learning_rate, batch_size)

        # Check if model exists first
        model_name = f"{architecture}_lr{learning_rate}_bs{batch_size}.h5"
        model_path = os.path.join("./model6", model_name)
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model tidak ditemukan di {model_path}. Harap periksa kembali konfigurasi model.")
            return
            
        if st.button("🔍 Prediksi"):
            with st.spinner("⏳ Sedang memproses gambar..."):
                try:
                    # Load model (sequence diagram: load_model)
                    model = self.model_handler.load_model(architecture, learning_rate, batch_size)
                    if not model:
                        st.error("❌ Gagal memuat model. Periksa konsol untuk informasi lebih lanjut.")
                        return
                    
                    # Preprocess image (sequence diagram: preprocess_image)
                    img_array, success = self.model_handler.preprocess_image(uploaded_file)
                    if not success:
                        st.error(f"❌ Gagal memproses gambar: {img_array}")
                        return

                    # Predict (sequence diagram: model.predict())
                    probabilities = model.predict(img_array)[0]
                    
                    # Process predictions (sequence diagram: process_predictions)
                    predicted_class, confidence = self.model_handler.process_predictions(probabilities)
                    
                    # Display prediction result (sequence diagram: display_prediction_result)
                    self.ui.display_prediction_result(predicted_class, confidence)
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")