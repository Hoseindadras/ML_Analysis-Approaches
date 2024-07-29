#ml_7
import os
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import zipfile

# Constants
IMG_SIZE = 150
ZIP_PATH = "/mnt/data/image.zip"
EXTRACTED_DIR = "/mnt/data/image/"

# Extract the ZIP
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACTED_DIR)

# Load and preprocess images
def load_and_preprocess_images(image_dir):
    images = []
    labels = []
    filenames = os.listdir(image_dir)
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        if filename.startswith("j"):
            labels.append(1)  # forest
        elif filename.startswith("s"):
            labels.append(0)  # sea
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels

# Extract features using HOG
def extract_features(image):
    gray_image = rgb2gray(image)
    features, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True, multichannel=False)
    return features

# Load data
images, labels = load_and_preprocess_images(EXTRACTED_DIR)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train_features = np.array([extract_features(img) for img in X_train])
X_test_features = np.array([extract_features(img) for img in X_test])

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_features, y_train)
y_pred = clf.predict(X_test_features)

# Results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Sea", "Forest"])

# Save the confusion matrix visualization
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix")
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(["Sea", "Forest"]))
plt.xticks(tick_marks, ["Sea", "Forest"], rotation=45)
plt.yticks(tick_marks, ["Sea", "Forest"])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(EXTRACTED_DIR + "confusion_matrix.png")

# Save misclassified images
misclassified_images = X_test[y_test != y_pred]
for idx, img in enumerate(misclassified_images):
    path = EXTRACTED_DIR + f"misclassified_image_{idx}.png"
    plt.imsave(path, img)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
