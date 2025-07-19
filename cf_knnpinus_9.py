import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# === Data dari soal 6 ===
circumference = [0.3, 0.18, 0.46, 0.63, 0.23, 0.56, 0.39, 0.41, 0.62, 0.43,
                 0.15, 0.19, 0.17, 0.17, 0.22, 0.45, 0.39, 0.42, 0.38, 0.3, 0.18]
height = [26.1, 21.51, 8.83, 12.08, 5.81, 13.5, 10.9, 6.79, 10.66, 10.5,
          2.67, 20.34, 19.72, 19.8, 23.7, 32.51, 26.23, 32.51, 29.18, 26.1, 21.51]
labels = ["White Pine", "White Pine", "Douglas Fir", "Douglas Fir", "Douglas Fir",
          "Douglas Fir", "Douglas Fir", "Douglas Fir", "Douglas Fir", "Douglas Fir",
          "Douglas Fir", "White Pine", "White Pine", "White Pine", "White Pine",
          "White Pine", "White Pine", "White Pine", "White Pine", "White Pine", "White Pine"]

# Hitung diameter
diameter = np.array(circumference) / np.pi

# Dataset
df = pd.DataFrame({
    "Diameter": diameter,
    "Tinggi": height,
    "Jenis": labels
})

# === Certainty Factor Function ===
def certainty_factor(d, t):
    # Aturan pakar sederhana
    cf_df = min(0.8 if d < 0.1 else 0.3, 0.3 if t > 20 else 0.7)
    cf_wp = min(0.2 if d < 0.1 else 0.7, 0.9 if t > 20 else 0.3)
    if cf_df > cf_wp:
        return "Douglas Fir", cf_df, cf_wp
    else:
        return "White Pine", cf_df, cf_wp

# === Model KNN ===
X = df[["Diameter", "Tinggi"]]
y = df["Jenis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# === Prediksi Gabungan CF + KNN ===
def predict_and_validate(diameter_input, tinggi_input):
    cf_result, cf_df, cf_wp = certainty_factor(diameter_input, tinggi_input)
    knn_pred = knn.predict([[diameter_input, tinggi_input]])[0]
    valid = (cf_result == knn_pred)
    
    print("=== HASIL PREDIKSI SISTEM CERDAS ===")
    print(f"Input Diameter: {diameter_input}")
    print(f"Input Tinggi  : {tinggi_input}")
    print(f"[CF] Prediksi Pakar     : {cf_result}")
    print(f"[CF] Skor Douglas Fir   : {cf_df}")
    print(f"[CF] Skor White Pine    : {cf_wp}")
    print(f"[KNN] Prediksi ML       : {knn_pred}")
    if valid:
        print(f"✅ Hasil Final: {cf_result} (divalidasi oleh KNN)")
    else:
        print(f"⚠️  Hasil Final: {cf_result} (bertentangan dengan KNN, perlu verifikasi)")

# === Contoh Eksekusi ===
predict_and_validate(0.095, 26.1)  # diameter dan tinggi input dari user
