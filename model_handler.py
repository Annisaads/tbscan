import streamlit as st
import os
import tensorflow as tf
import numpy as np

class ModelHandler:
    """Kelas untuk menangani pemilihan dan penggunaan model."""
    
    def __init__(self):
        """Inisialisasi ModelHandler."""
        if "config_saved" not in st.session_state:
            st.session_state["config_saved"] = False
            st.session_state["architecture"] = None
            st.session_state["learning_rate"] = None
            st.session_state["batch_size"] = None
    
    def sidebar_konfigurasi(self):
        """Fungsi untuk menangani pengaturan model di sidebar - sesuai sequence diagram."""
        st.sidebar.header("Konfigurasi Model")
        architecture = st.sidebar.selectbox("Pilih Arsitektur Model:", ["", "VGG19", "ResNet50", "DenseNet121"])
        learning_rate = st.sidebar.selectbox("Pilih Learning Rate:", ["", 0.01, 0.001])
        batch_size = st.sidebar.selectbox("Pilih Batch Size:", ["", 32, 64])

        if st.sidebar.button("Simpan Konfigurasi"):
            if architecture and learning_rate and batch_size:
                # Simpan konfigurasi model - sesuai sequence diagram
                st.session_state.update({
                    "architecture": architecture,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "config_saved": True
                })
                st.sidebar.markdown('<div class="success-message">Konfigurasi berhasil disimpan.</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div class="warning-message">Mohon lengkapi semua konfigurasi.</div>', unsafe_allow_html=True)

        return st.session_state["architecture"], st.session_state["learning_rate"], st.session_state["batch_size"]
    
    def load_model(self, architecture, learning_rate, batch_size):
        """Muat model berdasarkan konfigurasi yang dipilih - sesuai sequence diagram."""
        if not all([architecture, learning_rate, batch_size]):
            return None
            
        model_name = f"{architecture}_lr{learning_rate}_bs{batch_size}.h5"
        model_path = os.path.join("./model6", model_name)
        
        if os.path.exists(model_path):
            try:
                # Load model - sequence diagram load_model()
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,  # Set compile=False to avoid issues with optimizer
                    custom_objects=None
                )
                return model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                
                # Try with an alternative loading method
                try:
                    # Clear Keras backend session
                    tf.keras.backend.clear_session()
                    
                    # Load with custom_objects and safe mode
                    custom_objects = {}
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        custom_objects=custom_objects
                    )
                    return model
                except Exception as nested_e:
                    st.error(f"Failed to load model with alternative method: {str(nested_e)}")
                    return None
        else:
            return None
    
    def preprocess_image(self, image_path):
        """Proses gambar sebelum diprediksi - sesuai sequence diagram."""
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            return np.expand_dims(img_array, axis=0), True
        except Exception as e:
            return str(e), False
    
    def process_predictions(self, probabilities):
        """Proses hasil prediksi dan tentukan kelas - sesuai sequence diagram."""
        # Convert numpy array to scalar value if needed
        if isinstance(probabilities, np.ndarray):
            probabilities = float(probabilities.item()) if probabilities.size == 1 else float(probabilities[0])
        else:
            probabilities = float(probabilities)
            
        threshold = 0.1
        predicted_class = int(probabilities > threshold)
        confidence = probabilities if predicted_class == 1 else (1 - probabilities)
        
        return predicted_class, confidence