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
        
train_images, train_labels = make_test_data("/Users/mjas0/OneDrive/Desktop/cancer/photos/Dataset_BUSI_with_GT")
train_x = np.array(train_images)
train_y = np.array(train_labels)
mean_pixel_intensity = [np.mean(image) for image in train_x]

plt.scatter(mean_pixel_intensity, train_y.flatten())
plt.title('Mean Pixel Intensity vs. Class')
plt.xlabel('Class')
plt.ylabel('Mean Pixel Intensity')
#plt.show()

train_x = train_x.reshape(-1,1)
train_y = train_y.reshape(-1,1)
min_size = 1560
train_x = train_x[:min_size]
train_y = train_y[:min_size]
print(train_x.shape[0], train_y.shape[0])
train_y = train_y.ravel()

#Testing data
test_images, test_labels = make_test_data("/Users/mjas0/OneDrive/Desktop/cancer/photos/testing")
test_y = np.array(test_labels)
test_x = np.array(test_images)
print(test_labels)

test_x = test_x.reshape(-1,1)
test_y = test_y.reshape(-1,1)
min_size = 28
test_x = test_x[:min_size]
test_y = test_y[:min_size]
print(test_x.shape[0], test_y.shape[0])
test_x = test_x.ravel()


#Step 1: Train a k-NN model (choose some reasonable value for k)
k = 35  # Choose a value for k 
knn_model = KNeighborsClassifier(n_neighbors=k)
test_x = test_x.reshape(-1,1)
print("test x shape: ", test_x.shape) 
knn_model.fit(train_x, train_y)
#print("train_x:", train_x)
#TODO: Step 2: Make Predictions
y_pred = knn_model.predict(test_x) 

print("y_pred: ", y_pred)
    
#TODO: Step 3: Evaluate Model
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