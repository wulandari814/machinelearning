import cv2 # Ini Modulnya
import numpy as np

def extract_image_features(image_path):
    # Baca gambar dan resize
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))

    # Konversi ke grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Histogram intensitas piksel sebagai fitur dasar
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Ekstrak fitur tekstur sederhana (mean, std)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    return {
        "histogram": hist,
        "mean_gray": mean_intensity,
        "std_gray": std_intensity
    }
