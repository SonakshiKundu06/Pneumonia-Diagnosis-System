from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = tf.keras.models.load_model("trained_pneumonia_disease_model.keras")

# Load and convert original image to RGB
img_path = 'chest_xray//test//PNEUMONIA//person127_bacteria_603.jpeg'
original_img = Image.open(img_path).convert("RGB")  # ✅ Ensure RGB

# Resize for model input
img_resized = original_img.resize((128, 128))
img_array = img_to_array(img_resized)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
confidence = np.max(pred) * 100
predicted_class = ['NORMAL', 'PNEUMONIA'][np.argmax(pred)]

# Output
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# Show image
plt.imshow(original_img)
plt.axis('off')
plt.title(f"{predicted_class} ({confidence:.2f}%)", fontsize=14, color='green' if predicted_class == 'NORMAL' else 'red')
plt.show()