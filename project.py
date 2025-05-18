import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from zipfile import ZipFile
from collections import defaultdict
from scipy.ndimage import median_filter, binary_closing, generate_binary_structure
from scipy.fft import fft
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def rgb2gray(img):
    if len(img.shape) == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        return img

def apply_median_filter(img, size=3):
    return median_filter(img, size=size)

def rimlv_operator(gray_img, R=1, P=8):
    rows, cols = gray_img.shape
    angles = np.linspace(0, 2*np.pi, P, endpoint=False)
    offsets = [(int(round(R*np.cos(a))), int(round(R*np.sin(a)))) for a in angles]
    rimlv_img = np.zeros_like(gray_img, dtype=np.float32)

    for r in range(R, rows - R):
        for c in range(R, cols - R):
            gp = []
            for dx, dy in offsets:
                gp.append(gray_img[r + dy, c + dx])
            gp = np.array(gp, dtype=np.float32)
            mu = gray_img[r, c]
            variance = np.sum((gp - mu)**2) / P
            rimlv_img[r, c] = variance
    rimlv_img = cv2.normalize(rimlv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return rimlv_img

def threshold_image(img, thresh_val=None):
    if thresh_val is None:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return binary

def subtract_reference(def_img, ref_img):
    diff = cv2.absdiff(def_img, ref_img)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return diff

def morphological_closing(binary_img, structure_size=3):
    structure = generate_binary_structure(2, 1)
    for _ in range(structure_size//2):
        binary_img = binary_closing(binary_img, structure=structure)
    return binary_img.astype(np.uint8) * 255

def extract_fourier_descriptors(contour, num_descriptors=32):
    contour = contour[:, 0, :]
    cx = np.mean(contour[:, 0])
    cy = np.mean(contour[:, 1])
    Zn = (contour[:, 0] - cx) + 1j * (contour[:, 1] - cy)
    ak = fft(Zn)
    ak_mag = np.abs(ak)
    if ak_mag[0]==0:
        normalized = ak_mag[:num_descriptors]
    else:
        normalized = ak_mag[:num_descriptors] / ak_mag[0]
    return normalized.real

def detect_defects(image, reference_image):
    gray = rgb2gray(image)
    gray_filtered = apply_median_filter(gray, size=3)
    rimlv_img = rimlv_operator(gray_filtered, R=1, P=8)
    thresh = threshold_image(rimlv_img)
    ref_gray = rgb2gray(reference_image)
    ref_gray_filtered = apply_median_filter(ref_gray, size=3)
    ref_rimlv = rimlv_operator(ref_gray_filtered, R=1, P=8)
    ref_thresh = threshold_image(ref_rimlv)
    diff = subtract_reference(thresh, ref_thresh)
    closed = morphological_closing(diff, structure_size=3)
    return closed

def extract_features_from_mask(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    features = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            fd = extract_fourier_descriptors(cnt, num_descriptors=32)
            features.append(fd)
    return features

def load_images_from_zip(zip_file):
    images = defaultdict(list)  # label -> list of imgs
    with ZipFile(zip_file) as zipf:
        namelist = zipf.namelist()
        for name in namelist:
            if name.endswith(('.png','.jpg','.jpeg','.bmp')):
                # Expect folder structure: label/imgname.jpg
                parts = name.split('/')
                if len(parts) < 2:
                    continue
                label = parts[-2]
                with zipf.open(name) as file:
                    img_bytes = file.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        images[label].append(img)
    return images

st.title("Ceramic Tile Surface Defect Detection and Classification")

st.markdown("""
Upload a reference defect-free tile image first, then upload a zip file containing your dataset organized by defect type folders (e.g., CrackI/, CrackII/, PinHole/, Hole/, no_defect/).  
The app will detect defects, extract features, train a classifier, and show classification results.
""")

ref_img_file = st.file_uploader("Upload Reference Defect-Free Tile Image (jpg/png)", type=['png','jpg','jpeg'])
dataset_zip_file = st.file_uploader("Upload Dataset ZIP file with folders for each class", type=['zip'])

if ref_img_file is not None and dataset_zip_file is not None:
    ref_bytes = np.frombuffer(ref_img_file.read(), np.uint8)
    reference_image = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Loading dataset and extracting features..."):
        images_dict = load_images_from_zip(dataset_zip_file)
        st.success(f"Found classes: {list(images_dict.keys())}")

        X, y = [], []
        label_map = {}
        label_inv_map = {}
        label_idx = 0
        for label in sorted(images_dict.keys()):
            label_map[label] = label_idx
            label_inv_map[label_idx] = label
            label_idx += 1

        defect_free_label = None
        if 'no_defect' in label_map:
            defect_free_label = label_map['no_defect']

        for label, imgs in images_dict.items():
            st.write(f"Processing '{label}' with {len(imgs)} images")
            for img in imgs:
                defect_mask = detect_defects(img, reference_image)
                feats = extract_features_from_mask(defect_mask)
                if len(feats) == 0:
                    if label == 'no_defect':
                        X.append(np.zeros(32))
                        y.append(label_map[label])
                    else:
                        # no features detected for defect class, print warning
                        st.warning(f"No defect detected for an image in class '{label}' - skipping.")
                    continue
                for f in feats:
                    X.append(f)
                    y.append(label_map[label])

        if len(X) == 0:
            st.error("No features extracted from dataset. Check images and reference tile.")
        else:
            X = np.array(X)
            y = np.array(y)

            # Optionally exclude defect-free samples from training if desired
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            clf = SVC(kernel='rbf', decision_function_shape='ovr')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Classification Accuracy: **{accuracy*100:.2f}%**")

            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6,5))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            fig.colorbar(cax)
            ax.set_xticks(range(len(label_map)))
            ax.set_yticks(range(len(label_map)))
            ax.set_xticklabels([label_inv_map[i] for i in range(len(label_map))], rotation=45)
            ax.set_yticklabels([label_inv_map[i] for i in range(len(label_map))])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)

            # Show some defect masks from test data for demonstration
            st.subheader("Sample Defect Masks from Dataset")
            count = 0
            max_show = 4
            for label, imgs in images_dict.items():
                if count >= max_show:
                    break
                for img in imgs:
                    defect_mask = detect_defects(img, reference_image)
                    # Show original and mask side by side
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask_gray = defect_mask
                    fig2, axs = plt.subplots(1,2, figsize=(8,4))
                    axs[0].imshow(img_rgb)
                    axs[0].set_title(f"Original Image - {label}")
                    axs[0].axis('off')
                    axs[1].imshow(mask_gray, cmap='gray')
                    axs[1].set_title("Detected Defect Mask")
                    axs[1].axis('off')
                    st.pyplot(fig2)
                    count += 1
                    if count >= max_show:
                        break

else:
    st.info("Please upload both reference image and dataset zip file to start.")

