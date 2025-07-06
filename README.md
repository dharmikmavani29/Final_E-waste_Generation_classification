#  E-Waste Image Classification with EfficientNetV2B0 - Full Project README

---

#  End-to-End Deep Learning Pipeline with Streamlit Integration

---

## 📂 Dataset Access

The dataset required to run this project is available on Kaggle:

[🔗 E-Waste Image Dataset on Kaggle](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)

After downloading, extract the dataset and update these paths in your code accordingly:

```python
testpath = r'path-to-dataset/test'
trainpath = r'path-to-dataset/train'
validpath = r'path-to-dataset/val'
```

---

## 🗂️ **1. Import Libraries**

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
```

✅ **Purpose:**

* TensorFlow for model building
* NumPy for array operations
* Matplotlib/Seaborn for visualization
* Sklearn for evaluation metrics

---

## 📂 **2. Dataset Loading**

```python
testpath = r'E:\AICTE INTERNSHIP\modified-dataset\test'
trainpath = r'E:\AICTE INTERNSHIP\modified-dataset\train'
validpath = r'E:\AICTE INTERNSHIP\modified-dataset\val'

# Load datasets
datavalid = tf.keras.utils.image_dataset_from_directory(validpath, shuffle=True, batch_size=32, image_size=(128, 128))
datatest = tf.keras.utils.image_dataset_from_directory(testpath, shuffle=False, batch_size=32, image_size=(128, 128))
datatrain = tf.keras.utils.image_dataset_from_directory(trainpath, shuffle=True, batch_size=32, image_size=(128, 128))

class_names = datatrain.class_names
print(len(class_names))
print(class_names)
```

✅ **Purpose:**

* Loads images for training, validation, and testing
* Automatically assigns class labels

---

## 🖼️ **3. Sample Visualization**

```python
plt.figure(figsize=(10, 10))
for images, labels in datatrain.take(1):
    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
```

✅ **Visual check:** Ensures correct data loading and class distribution.

---

## 📊 **4. Class Distribution Plot Function**

```python
def plot_class_distribution(dataset, title="Class Distribution"):
    class_counts = {}
    for images, labels in dataset:
        for label in labels.numpy():
            class_name = dataset.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Number of Items")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
```

✅ **Purpose:** Visualizes dataset balance for each class.

---

## 🔁 **5. Data Augmentation Pipeline**

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
```

✅ **Why:**

* Simulates real-world image variations
* Improves model generalization

---

## 🏗️ **6. Transfer Learning with EfficientNetV2B0**

```python
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
for layer in base_model.layers[:97]:
    layer.trainable = False
```

✅ **Why:**

* Uses pretrained ImageNet knowledge
* Only fine-tunes deeper layers to avoid overfitting

---

## 🧩 **7. Full Model Assembly**

```python
NUM_CLASSES = 10
model = tf.keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
```

✅ Combines base model with custom classifier layers.

---

## ⚙️ **8. Compile Model & Callbacks**

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000101),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['Accuracy'])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_lr=1e-6)
```

✅ **Purpose:**

* Prevents overfitting with Early Stopping
* Dynamically reduces learning rate on plateaus

---

## 🚀 **9. Train the Model**

```python
history = model.fit(
    datatrain,
    validation_data=datavalid,
    epochs=20,
    callbacks=[early, reduce_lr]
)
```

---

## 📈 **10. Visualize Accuracy and Loss Curves**

```python
acc = history.history['Accuracy']
val_acc = history.history['val_Accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
```

---

## 🧪 **11. Evaluate on Test Data**

```python
loss, accuracy = model.evaluate(datatest)
print(f'Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}')
```

---

## 🧩 **12. Confusion Matrix and Classification Report**

```python
y_true = np.concatenate([y.numpy() for x, y in datatest], axis=0)
y_pred_probs = model.predict(datatest)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
print(cm)
print(classification_report(y_true, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

---

## 💾 **13. Save the Trained Model**

```python
model.save('best_model.keras')
```

---

# 🌐 **14. Streamlit App (app.py)**

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-Waste Image Classifier", layout="centered")
st.title("\u267b\ufe0f E-Waste Image Classification App")

class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

model = load_model()

uploaded_files = st.file_uploader("Upload E-Waste Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
        img_array = np.expand_dims(np.array(img).astype(np.float32), axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"\ud83c\udff7\ufe0f Predicted Class: **{predicted_class}**")
        st.info(f"\ud83d\udd25 Confidence: {confidence:.2f}%")

        fig, ax = plt.subplots()
        ax.barh(class_names, prediction[0] * 100, color='cyan')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Class-wise Confidence Distribution")
        st.pyplot(fig)
```

---

## 🚀 **15. Run the App**

```bash
streamlit run app.py
```

**App Usage:**

* Upload images (jpg, png)
* Get instant class prediction with confidence
* Visual bar chart for probability distribution

---

# ✅ **End-to-End E-Waste Classifier with Visualization and Deployment Completed**

---
