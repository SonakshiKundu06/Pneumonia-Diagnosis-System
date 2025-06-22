import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore


model = tf.keras.models.load_model("trained_pneumonia_disease_model.keras")

 #Test Image Visualization
import cv2
image_path = 'C:\\Users\\insp 5430\\programmings\\datascience\\chest_xray\\test\\PNEUMONIA\\person1_virus_7.jpeg'
# Reading an image in default mode
img = cv2.imread(image_path)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
# Displaying the image 
plt.imshow(img1)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)

input_arr = preprocess_input(input_arr)  # ✅ This line is MANDATORY
input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch

predictions = model.predict(input_arr)

result_index = np.argmax(predictions) #Return index of max element
print(result_index)

class_name=['NORMAL','PNEUMONIA']

model_prediction=class_name[result_index]
plt.imshow(img1)
plt.title(f"Condition: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()