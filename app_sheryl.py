import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Deteksi Obesitas",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Kustom untuk Tampilan UI yang Lebih Baik ---
st.markdown("""
<style>
   .stApp {
        background: linear-gradient(135deg, #0C551F 0%, #E6D4E6 100%); /* Gradasi dari ocean ke lime */
        color: #ffffff; /* Putih untuk teks agar kontras dengan latar */
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        color: #D8E0A5; /* Lime untuk judul utama */
        text-align: center;
        font-size: 3em;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        color: #ffffff; /* Putih untuk subjudul */
        font-size: 2em;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #D8E0A5; /* Lime untuk border */
    }
    h3 {
        color: #ffffff; /* Putih untuk subjudul kecil */
        text-align: center;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stForm {
        background: ##7ed957 ; 
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
    }
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1.1em;
        font-weight: semibold;
        color: #ffffff; /* Putih */
        margin-bottom: 8px;
    }
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
        font-size: 1.1em;
        font-weight: semibold;
        color: #ffffff; /* Putih */
        margin-bottom: 8px;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput div[data-baseweb="input"]:hover {
        border-color: #ffffff; /* Lime pada hover */
        background-color: #ffffff; /* Ocean pada hover */
        color: #ffffff; /* Putih untuk teks saat hover */
    }
    .stSelectbox div[data-baseweb="select"] input, .stNumberInput div[data-baseweb="input"] input {
        color: #1F485D !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1F485D 0%, #D8E0A5 100%); /* Gradasi ocean ke lime */
        color: #ffffff;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #173a4a 0%, #c1d08d 100%); /* Variasi lebih gelap dan terang */
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .stDataFrame table {
        background-color: #D8E0A5; /* Lime */
        color: #1F485D; /* Ocean */
    }
    .stDataFrame th {
        background-color: #1F485D; /* Ocean */
        color: #ffffff;
        font-weight: bold;
    }
    .stDataFrame td {
        border-top: 1px solid #1F485D; /* Ocean untuk border */
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #ffffff;
        font-weight: bold;
    }
    .stSuccess {
        background: linear-gradient(90deg, #1F485D 0%, #D8E0A5 100%); /* Gradasi ocean ke lime */
    }
    .stError {
        background-color: #ff6666;
    }
    .stWarning {
        background-color: #ffd700;
        color: #2c3e50;
    }
    .css-1offfwp {
        padding: 0 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    try:
        all_tuned_models = joblib.load('models/all_tuned_models.pkl')
        model = all_tuned_models['Random Forest']
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = label_encoders['NObeyesdad']
        return model, scaler, label_encoders, target_encoder
    except Exception as e:
        st.error(f"Gagal memuat model/preprocessing: {e}")
        st.stop()

def preprocess_input(input_df, label_encoders, scaler):
    categorical_cols = [col for col in label_encoders.keys() if col != 'NObeyesdad']
    
    for col in categorical_cols:
        if col in input_df.columns:
            if input_df[col].iloc[0] not in label_encoders[col].classes_:
                st.warning(f"Nilai tidak dikenal di kolom {col}.")
                input_df[col] = pd.Categorical(input_df[col], categories=label_encoders[col].classes_).codes[0]
            else:
                input_df[col] = label_encoders[col].transform(input_df[col])

    if hasattr(scaler, 'feature_names_in_'):
        input_df = input_df[scaler.feature_names_in_]
    
    try:
        return scaler.transform(input_df)
    except Exception as e:
        st.error(f"Gagal preprocessing data: {e}")
        st.stop()

# Validasi helper
def to_float(val, field):
    try:
        return float(val)
    except:
        st.error(f"Input '{field}' harus berupa angka.")
        st.stop()

def validate_inputs(age, height, weight, fcvc, ncp, ch2o, faf, tue):
    if not (10 <= age <= 100):
        st.error("Usia harus antara 10 dan 100 tahun.")
        st.stop()
    if not (1.0 <= height <= 2.5):
        st.error("Tinggi harus antara 1.0 dan 2.5 meter.")
        st.stop()
    if not (30 <= weight <= 300):
        st.error("Berat harus antara 30 dan 300 kg.")
        st.stop()
    if not (1 <= fcvc <= 3):
        st.error("Frekuensi Konsumsi Sayuran harus antara 1 dan 3.")
        st.stop()
    if not (1 <= ncp <= 4):
        st.error("Jumlah Makan Utama harus antara 1 dan 4.")
        st.stop()
    if not (0 <= ch2o <= 5):
        st.error("Konsumsi Air harus antara 0 dan 5 liter.")
        st.stop()
    if not (0 <= faf <= 3):
        st.error("Frekuensi Aktivitas Fisik harus antara 0 dan 3.")
        st.stop()
    if not (0 <= tue <= 24):
        st.error("Waktu Teknologi harus antara 0 dan 24 jam.")
        st.stop()

def main():
    st.markdown("<h1>Prediksi Tingkat Obesitas</h1>", unsafe_allow_html=True)

    # Panduan Penggunaan
    with st.expander("Panduan Penggunaan"):
        st.markdown("""
        **Cara Menggunakan Aplikasi:**
        1. Masukkan data Anda pada formulir di atas.
        2. Pastikan semua input berupa angka untuk kolom numerik.
        3. Klik tombol 'Prediksi' untuk melihat hasil.

        **Catatan:**
        - Model ini hanya memberikan perkiraan berdasarkan data input.
        - Konsultasikan dengan dokter untuk diagnosis resmi.
        """)

    with st.form("prediction_form"):
        st.markdown("<h3>Input Data Pasien</h3>", unsafe_allow_html=True)

        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        age = to_float(st.text_input("Usia (tahun)", "25"), "Usia")
        height = to_float(st.text_input("Tinggi (meter)", "1.80"), "Tinggi")
        weight = to_float(st.text_input("Berat (kg)", "85.0"), "Berat")
        family_history = st.selectbox("Riwayat Keluarga Obesitas", ['yes', 'no'])
        favc = st.selectbox("Sering Makan Tinggi Kalori", ['yes', 'no'])
        fcvc = to_float(st.text_input("Frekuensi Konsumsi Sayuran (1-3)", "2.0"), "FCVC")
        ncp = to_float(st.text_input("Jumlah Makan Utama (1-4)", "3.0"), "NCP")
        caec = st.selectbox("Konsumsi Antar Waktu Makan", ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.selectbox("Merokok", ['yes', 'no'])
        ch2o = to_float(st.text_input("Konsumsi Air (liter)", "2.0"), "CH2O")
        scc = st.selectbox("Monitor Konsumsi Kalori", ['yes', 'no'])
        faf = to_float(st.text_input("Frekuensi Aktivitas Fisik", "1.0"), "FAF")
        tue = to_float(st.text_input("Waktu Teknologi (jam)", "1.0"), "TUE")
        calc = st.selectbox("Konsumsi Alkohol", ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox("Transportasi Utama", ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        validate_inputs(age, height, weight, fcvc, ncp, ch2o, faf, tue)
        input_data = {
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history_with_overweight': [family_history], 'FAVC': [favc], 'FCVC': [fcvc],
            'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc],
            'FAF': [faf], 'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
        }
        input_df = pd.DataFrame(input_data)

        model, scaler, label_encoders, target_encoder = load_models()
        input_scaled = preprocess_input(input_df, label_encoders, scaler)

        if input_scaled is not None:
            prediction_encoded = model.predict(input_scaled)
            prediction_label = target_encoder.inverse_transform(prediction_encoded)

            # Kalkulator BMI
            bmi = weight / (height ** 2)
            bmi_category = {
                (0, 18.5): 'Kekurangan Berat Badan',
                (18.5, 25): 'Normal',
                (25, 30): 'Kelebihan Berat Badan',
                (30, float('inf')): 'Obesitas'
            }
            category = next(v for k, v in bmi_category.items() if k[0] <= bmi < k[1])
            st.markdown(f"<div class='stSuccess'>BMI Anda: <b>{bmi:.2f}</b> ({category})</div>", unsafe_allow_html=True)

            # Visualisasi Hasil (jika model mendukung predict_proba)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
                prob_df = pd.DataFrame({
                    'Tingkat Obesitas': target_encoder.classes_,
                    'Probabilitas': probabilities
                })
                fig = px.bar(prob_df, x='Tingkat Obesitas', y='Probabilitas', title='Distribusi Probabilitas Tingkat Obesitas')
                st.plotly_chart(fig)

            # Penjelasan Hasil
            obesity_explanations = {
                'Normal_Weight': 'Berat badan Anda dalam kisaran normal. Pertahankan pola hidup sehat!',
                'Overweight_Level_I': 'Anda berada pada tingkat kelebihan berat badan I. Pertimbangkan diet seimbang dan olahraga.',
                'Obesity_Type_I': 'Anda berada pada tingkat obesitas tipe I. Konsultasikan dengan dokter untuk rencana penurunan berat badan.',
                'Underweight': 'Berat badan Anda berada di bawah normal. Pertimbangkan konsultasi dengan ahli gizi untuk meningkatkan asupan nutrisi.',
                'Overweight_Level_II': 'Anda berada pada tingkat kelebihan berat badan II. Dianjurkan untuk segera mengubah pola makan dan meningkatkan aktivitas fisik.',
                'Obesity_Type_II': 'Anda berada pada tingkat obesitas tipe II. Segera konsultasikan dengan dokter untuk penanganan medis yang tepat.',
                'Obesity_Type_III': 'Anda berada pada tingkat obesitas tipe III (morbida). Perhatian medis mendesak diperlukan untuk menangani kondisi ini.'
                }
            
            st.markdown(f"<h3>Penjelasan:</h3> {obesity_explanations.get(prediction_label[0], 'Tidak ada penjelasan tersedia.')}", unsafe_allow_html=True)

            # Hasil Prediksi
            st.markdown(f"<div class='stSuccess'>Hasil Prediksi: Tingkat Obesitas Anda adalah <b>{prediction_label[0]}</b></div>", unsafe_allow_html=True)
            st.write("Data Input Anda:", input_df)

            # Riwayat Prediksi
            if 'prediction_history' not in st.session_state:
                st.session_state['prediction_history'] = []
            st.session_state['prediction_history'].append({
                'Tanggal': pd.Timestamp.now(),
                'Prediksi': prediction_label[0],
                **input_data
            })
            st.markdown("<h2>Riwayat Prediksi</h2>", unsafe_allow_html=True)
            history_df = pd.DataFrame(st.session_state['prediction_history'])
            st.dataframe(history_df)

if __name__ == "__main__":
    main()