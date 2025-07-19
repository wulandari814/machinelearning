import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from vision_module import extract_image_features

# === Data Tanaman Pinus ===
circumference = [0.3, 0.18, 0.46, 0.63, 0.23, 0.56, 0.39, 0.41, 0.62, 0.43,
                 0.15, 0.19, 0.17, 0.17, 0.22, 0.45, 0.39, 0.42, 0.38, 0.3, 0.18]
height = [26.1, 21.51, 8.83, 12.08, 5.81, 13.5, 10.9, 6.79, 10.66, 10.5,
          2.67, 20.34, 19.72, 19.8, 23.7, 32.51, 26.23, 32.51, 29.18, 26.1, 21.51]
labels = ["White Pine", "White Pine", "Douglas Fir", "Douglas Fir", "Douglas Fir",
          "Douglas Fir", "Douglas Fir", "Douglas Fir", "Douglas Fir", "Douglas Fir",
          "Douglas Fir", "White Pine", "White Pine", "White Pine", "White Pine",
          "White Pine", "White Pine", "White Pine", "White Pine", "White Pine", "White Pine"]

diameter = np.array(circumference) / np.pi

df = pd.DataFrame({
    "Diameter": diameter,
    "Tinggi": height,
    "Jenis": labels
})

def certainty_factor(d, t):
    cf_df = min(0.8 if d < 0.1 else 0.3, 0.3 if t > 20 else 0.7)
    cf_wp = min(0.2 if d < 0.1 else 0.7, 0.9 if t > 20 else 0.3)
    return ("Douglas Fir" if cf_df > cf_wp else "White Pine", cf_df, cf_wp)

# Training KNN
X = df[["Diameter", "Tinggi"]]
y = df["Jenis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

def predict_system(diameter_input, tinggi_input, image_path=None):
    cf_result, cf_df, cf_wp = certainty_factor(diameter_input, tinggi_input)
    knn_pred = knn.predict([[diameter_input, tinggi_input]])[0]

    print("\n=== HASIL PREDIKSI ===")
    print(f"Diameter: {diameter_input:.3f}, Tinggi: {tinggi_input:.2f}")
    print(f"[CF] Prediksi: {cf_result} (DF={cf_df}, WP={cf_wp})")
    print(f"[KNN] Prediksi: {knn_pred}")

    if image_path:
        try:
            image_features = extract_image_features(image_path)
            mean_gray = image_features["mean_gray"]
            std_gray = image_features["std_gray"]
            print(f"[Vision] Mean Gray: {mean_gray:.2f}, Std Gray: {std_gray:.2f}")
        except FileNotFoundError as e:
            print(f"[Vision] ❌ Gagal membaca gambar: {e}")
        except Exception as e:
            print(f"[Vision] ⚠️ Terjadi kesalahan saat ekstraksi gambar: {e}")

    valid = (cf_result == knn_pred)
    if valid:
        print(f"✅ Final: {cf_result} (divalidasi oleh KNN)")
    else:
        print(f"⚠️ Final: {cf_result} (bertentangan dengan KNN, perlu verifikasi)")

# Eksekusi langsung (gambar opsional)
if __name__ == "__main__":
    # Contoh: tanpa gambar
    predict_system(0.095, 26.1)

    # Contoh: dengan gambar (jika ada)
    # predict_system(0.095, 26.1, "pine_tree.jpg")
