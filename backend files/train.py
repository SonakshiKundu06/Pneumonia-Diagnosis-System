import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation # type: ignore
from tensorflow.keras.layers import Input # type: ignore

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Function to apply VGG16-specific preprocessing
def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

training_set = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\insp 5430\\programmings\\datascience\\chest_xray\\train",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_set = tf.keras.utils.image_dataset_from_directory(
   "C:\\Users\\insp 5430\\programmings\\datascience\\chest_xray\\val" ,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

test_set = tf.keras.utils.image_dataset_from_directory(
   "C:\\Users\\insp 5430\\programmings\\datascience\\chest_xray\\test" ,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
# Apply VGG16 preprocessing
train_ds = training_set.map(preprocess)
val_ds = validation_set.map(preprocess)
test_ds = test_set.map(preprocess)

# Improve performance with caching & prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Load pre-trained VGG16 model without top layers
vgg_base = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

# Freeze the VGG16 layers to prevent training
for layer in vgg_base.layers:
    layer.trainable = False

# Build the new model
model = Sequential()

# Add VGG16 base
model.add(vgg_base)

# Model with data augmentation
model = Sequential()
model.add(Input(shape=(128, 128, 3)))
model.add(data_augmentation)         # 👈 Augmentation added here
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
training_history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Unfreeze the last 4 layers of VGG16 for fine-tuning
for layer in vgg_base.layers[-4:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_history = model.fit(train_ds, validation_data=val_ds, epochs=5)


plt.plot(training_history.history['accuracy'], label='Train Accuracy')
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.show()

# Function to extract features from dataset
def extract_features(dataset, model):
    features = []
    labels = []
    for images, label in dataset:
        images = preprocess_input(images.numpy())
        output = model.predict(images)
        output = output.reshape(output.shape[0], -1)  # Flatten the output
        features.append(output)
        labels.append(label.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Load pre-trained VGG16 without top layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

# Extract features using VGG16
X_train, y_train = extract_features(training_set, feature_model)
X_val, y_val = extract_features(validation_set, feature_model)

# Apply PCA
pca = PCA(n_components=100)  # Tune this value as needed
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

# Train a classifier (SVM here)
svm_clf = SVC(kernel='rbf', C=10)
svm_clf.fit(X_train_pca, np.argmax(y_train, axis=1))

# Evaluate on validation data
y_pred = svm_clf.predict(X_val_pca)
accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred)

print("Validation Accuracy after PCA + VGG16 features:", accuracy)

#Training set Accuracy
train_loss, train_acc = model.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = model.evaluate(validation_set)
print('Validation accuracy:', val_acc)

#Test set Accuracy
test_loss, test_acc = model.evaluate(test_ds)
print('Test accuracy:', test_acc)

model.save('trained_pneumonia_disease_model.keras')