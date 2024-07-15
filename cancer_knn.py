import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def make_test_data(path):
    """
    Reads images from the given path, resizes them to 500x500, and assigns labels based on folder names.
    
    Parameters:
    path (str): The root directory containing subfolders of images.
    
    Returns:
    tuple: Lists of images and their corresponding labels.
    """
    train_images = []
    train_labels = []
    for class_id, class_name in enumerate(sorted(os.listdir(path))):
        new_path = os.path.join(path, class_name)
        for images in os.listdir(new_path):
            image_path = os.path.join(new_path, images)
            train_image = cv2.imread(image_path)
            train_image = cv2.resize(train_image, (500, 500))
            if train_image is not None:
                train_images.append(train_image)
                train_labels.append(class_id)
    return train_images, train_labels

# Load training data
train_images, train_labels = make_test_data("/Users/mjas0/OneDrive/Desktop/cancer/photos/Dataset_BUSI_with_GT")
train_x = np.array(train_images)
train_y = np.array(train_labels)

# Calculate mean pixel intensity for each image
mean_pixel_intensity = [np.mean(image) for image in train_x]

# Scatter plot of mean pixel intensity vs. class labels
plt.scatter(mean_pixel_intensity, train_y.flatten())
plt.title('Mean Pixel Intensity vs. Class')
plt.xlabel('Class')
plt.ylabel('Mean Pixel Intensity')
# plt.show()  # Uncomment to display the plot

# Reshape training data for k-NN
train_x = train_x.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)

# Use a minimum size to ensure consistent data size
min_size = 1560
train_x = train_x[:min_size]
train_y = train_y[:min_size]
print(train_x.shape[0], train_y.shape[0])
train_y = train_y.ravel()  # Flatten the labels

# Load testing data
test_images, test_labels = make_test_data("/Users/mjas0/OneDrive/Desktop/cancer/photos/testing")
test_y = np.array(test_labels)
test_x = np.array(test_images)
print(test_labels)

# Reshape testing data for k-NN
test_x = test_x.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

# Use a minimum size to ensure consistent data size
min_size = 28
test_x = test_x[:min_size]
test_y = test_y[:min_size]
print(test_x.shape[0], test_y.shape[0])
test_x = test_x.ravel()  # Flatten the test images

# Step 1: Train a k-NN model (choose some reasonable value for k)
k = 35  # Choose a value for k 
knn_model = KNeighborsClassifier(n_neighbors=k)
test_x = test_x.reshape(-1, 1)  # Reshape the test data
print("test x shape: ", test_x.shape) 
knn_model.fit(train_x, train_y)

# Step 2: Make Predictions
y_pred = knn_model.predict(test_x)
print("y_pred: ", y_pred)

# Step 3: Evaluate Model
accuracy = accuracy_score(y_pred=y_pred, y_true=test_y)

# Step 4: Visualize the confusion matrix as a heatmap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf_matrix = confusion_matrix(y_true=test_y, y_pred=y_pred)
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn_model.classes_)
disp.plot(cmap='Reds')
plt.show()

print('----------------------------')
print(f"Accuracy: {np.around(accuracy*100, 2)}%")
