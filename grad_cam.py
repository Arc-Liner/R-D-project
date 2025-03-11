import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Load a pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess an image
img_path = './data/Brain_MRI/no1.jpeg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get the top predicted class
preds = model.predict(x)
predicted_class = np.argmax(preds[0])
pred_class_name = decode_predictions(preds)[0][0][1]

# Get the output tensor of the last convolutional layer
last_conv_layer = model.get_layer('block5_conv3')

# Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

# Compute the gradient of the top predicted class with respect to the output feature map of the last conv layer
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)[0]

# Compute the CAM
cam = np.mean(conv_outputs[0], axis=-1)
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()


# Generate heatmap overlay
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(
    cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB).astype('float32'), 0.6,
    heatmap.astype('float32'), 0.4, 0
)
# Plot the original image, Grad-CAM heatmap, and the superimposed image
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.title(f'Grad-CAM Heatmap ({pred_class_name})')
plt.imshow(heatmap)

plt.subplot(133)
plt.title(f'Superimposed Image ({pred_class_name})')
plt.imshow(superimposed_img)
plt.show()