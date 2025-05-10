import streamlit as st
import hydralit_components as hc
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== Konfigurasi Halaman =====================
st.set_page_config(page_title="TBScan", page_icon="🩺")

# ===================== Fungsi Utility =====================
def load_model(architecture, learning_rate, batch_size):
    """Muat model berdasarkan konfigurasi yang dipilih."""
    model_name = f"{architecture}_lr{learning_rate}_bs{batch_size}.h5"
    model_path = os.path.join("./model3", model_name)
    return tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None

def preprocess_image(image_path):
    """Proses gambar sebelum diprediksi."""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0), True
    except Exception as e:
        return str(e), False

def local_css(file_name):
    """Menambahkan CSS lokal."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ===================== Sidebar Konfigurasi =====================
def sidebar_konfigurasi():
    """Fungsi untuk menangani pengaturan model di sidebar."""
    if "config_saved" not in st.session_state:
        st.session_state["config_saved"] = False
        st.session_state["architecture"] = None
        st.session_state["learning_rate"] = None
        st.session_state["batch_size"] = None

    st.sidebar.header("Konfigurasi Model")
    architecture = st.sidebar.selectbox("Pilih Arsitektur Model:", ["", "VGG19", "ResNet50", "DenseNet121"])
    learning_rate = st.sidebar.selectbox("Pilih Learning Rate:", ["", 0.01, 0.001])
    batch_size = st.sidebar.selectbox("Pilih Batch Size:", ["", 32, 64])

    if st.sidebar.button("Simpan Konfigurasi"):
        if architecture and learning_rate and batch_size:
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

# ===================== Halaman Prediksi =====================
def prediction_page():
    """Fungsi untuk halaman prediksi tuberkulosis."""
    st.title("🔍 Prediksi Tuberkulosis pada Citra Chest X-ray")
    st.header("📂 Unggah Gambar X-ray")
    
    uploaded_file = st.file_uploader("Pilih gambar untuk diklasifikasikan", type=["jpg", "png", "jpeg"])
    if not uploaded_file:
        return

    st.image(uploaded_file, caption="🖼️ Gambar yang diunggah", use_container_width=True)
    architecture, learning_rate, batch_size = sidebar_konfigurasi()

    if not (architecture and learning_rate and batch_size):
        return  

    st.markdown("---")
    st.markdown("### ⚙️ Konfigurasi Aktif")
    st.info(f"Arsitektur: `{architecture}` | Batch Size: `{batch_size}` | Learning Rate: `{learning_rate}`")
    st.markdown("---")

    model = load_model(architecture, learning_rate, batch_size)
    if not model:
        st.error("❌ Model tidak ditemukan. Harap unggah model yang sesuai.")
        return

    if st.button("🔍 Prediksi"):
        with st.spinner("⏳ Sedang memproses gambar..."):
            try:
                img_array, _ = preprocess_image(uploaded_file)
                probabilities = model.predict(img_array)[0]
                predicted_class, tb_prob, normal_prob = process_predictions(probabilities)

                display_prediction_result(predicted_class, tb_prob, normal_prob)
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")

def process_predictions(probabilities):
    """Proses hasil prediksi dan hitung probabilitas."""
    if len(probabilities) == 1:
        tb_prob = probabilities[0] * 100
        normal_prob = (1 - probabilities[0]) * 100
        predicted_class = int(probabilities[0] > 0.5)
    else:
        predicted_class = np.argmax(probabilities)
        tb_prob = probabilities[1] * 100
        normal_prob = probabilities[0] * 100

    return predicted_class, tb_prob, normal_prob

def display_prediction_result(predicted_class, tb_prob, normal_prob):
    """Tampilkan hasil prediksi."""
    if predicted_class == 1:
        result_html = generate_prediction_html("Tuberkulosis", "🚨 Tuberkulosis", "Hasil prediksi menunjukkan kemungkinan adanya infeksi tuberkulosis pada paru-paru. Sebaiknya segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        result_html = generate_prediction_html("Normal", "✅ Normal", "Hasil prediksi menunjukkan bahwa paru-paru dalam kondisi sehat dan bebas dari tuberkulosis. Selalu perhatikan kesehatan Anda dan lakukan pemeriksaan secara rutin untuk menjaga kondisi paru-paru tetap baik.")

    st.markdown(result_html, unsafe_allow_html=True)
    st.subheader("📊 Probabilitas Tiap Kelas:")
    st.write(f"**Normal:** {normal_prob:.2f}%")
    st.write(f"**Tuberkulosis:** {tb_prob:.2f}%")

def generate_prediction_html(status, title, message):
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

# ===================== Halaman Evaluasi =====================
def evaluation_page():
    """Fungsi untuk halaman evaluasi model."""
    architecture, learning_rate, batch_size = sidebar_konfigurasi()

    if st.session_state.get("config_saved"):
        st.info(f"**Konfigurasi Aktif:**  \nArsitektur: `{st.session_state['architecture']}` | Batch Size: `{st.session_state['batch_size']}` | Learning Rate: `{st.session_state['learning_rate']}`", icon="⚙️")

    st.title("📈 Evaluasi Kinerja Model")
    st.write("Evaluasi ini menggunakan dataset default yang telah disediakan di dalam aplikasi.")

    dataset_root = "./dataset/evaluasi"
    normal_path, tb_path = os.path.join(dataset_root, "Normal"), os.path.join(dataset_root, "Tuberculosis")

    if not (os.path.exists(normal_path) and os.path.exists(tb_path)):
        st.error("Dataset evaluasi tidak ditemukan.")
        return

    if not (architecture and learning_rate and batch_size):
        return

    model = load_model(architecture, learning_rate, batch_size)
    if not model:
        st.error("Model tidak ditemukan.")
        return

    if st.button("🔎 Evaluasi"):
        dataset_images, true_labels = load_evaluation_data(normal_path, tb_path)
        if not dataset_images:
            st.error("Tidak ada data yang valid untuk diproses.")
            return

        evaluate_model(model, dataset_images, true_labels)

def load_evaluation_data(normal_path, tb_path):
    """Muat data evaluasi dari folder normal dan tuberkulosis."""
    dataset_images, true_labels = [], []
    for label, folder_path in enumerate([normal_path, tb_path]):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img_array, valid = preprocess_image(img_path)
            if valid:
                dataset_images.append(img_array)
                true_labels.append(label)
    return dataset_images, true_labels

def evaluate_model(model, dataset_images, true_labels):
    """Evaluasi model dan tampilkan hasil."""
    dataset_images = np.vstack(dataset_images)
    true_labels = np.array(true_labels)

    predictions = model.predict(dataset_images)
    pred_labels = (predictions > 0.5).astype(int).flatten() if predictions.shape[1] == 1 else np.argmax(predictions, axis=1)

    class_names = ["Normal", "Tuberkulosis"]
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

    display_evaluation_results(report, true_labels, pred_labels)

def display_evaluation_results(report, true_labels, pred_labels):
    """Tampilkan hasil evaluasi seperti confusion matrix dan laporan."""
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Tuberkulosis"], yticklabels=["Normal", "Tuberkulosis"], ax=ax)
    st.pyplot(fig)

    st.subheader("📋 Laporan Evaluasi:")
    st.write(f"Accuracy: {report['accuracy'] * 100:.2f}%")
    st.write(f"Precision: {report['Tuberkulosis']['precision'] * 100:.2f}%")
    st.write(f"Recall: {report['Tuberkulosis']['recall'] * 100:.2f}%")
    st.write(f"F1-score: {report['Tuberkulosis']['f1-score'] * 100:.2f}%")

# ===================== Fungsi Main =====================
def main():
    """Fungsi utama aplikasi."""
    st.markdown("<h1 style='text-align: center; color: #001B48;'>🩺 TBScan</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Aplikasi ini dirancang untuk memprediksi citra Chest X-Ray yang menunjukkan kondisi normal atau tuberkulosis, serta menampilkan evaluasi kinerja model klasifikasi tuberkulosis berdasarkan citra Chest X-Ray yang telah melalui proses pelatihan sebelumnya.</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Prediksi", use_container_width=True):
            st.session_state["page"] = "prediksi"
    with col2:
        if st.button("📈 Evaluasi", use_container_width=True):
            st.session_state["page"] = "evaluasi"

    if "page" not in st.session_state:
        st.session_state["page"] = "prediksi"

    if st.session_state["page"] == "prediksi":
        prediction_page()
    elif st.session_state["page"] == "evaluasi":
        evaluation_page()

if __name__ == "__main__":
    main()
