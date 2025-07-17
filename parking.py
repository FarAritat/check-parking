import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

input_dir = 'clf-data'
catelog = ['empty', 'not_empty']

data = []
labels = []
for cat_idx, category in enumerate(catelog):
    for file in os.listdir(os.path.join(input_dir, category)):
        file_path = os.path.join(input_dir, category, file)
        #print(f'Processing file: {file_path}')
        imread_img = imread(file_path)
        imread_img = resize(imread_img, (15, 15))  # Resize to 15x15
        data.append(imread_img.flatten())
        labels.append(cat_idx)

train_x, test_x, train_y, test_y = train_test_split(
    np.array(data), np.array(labels), test_size=0.2, random_state=42
)

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x, train_y)

y_pred = model.predict(test_x)
"""print(f'Test labels: {test_y}')
print(f'Predicted labels: {y_pred}')"""
accuracy = np.mean(y_pred == test_y)
print(f'Accuracy: {accuracy * 100:.2f}%')

#test
test_image_path = r'clf-data\not_empty\00000000_00000056.jpg'
test_image = imread(test_image_path)
test_image = resize(test_image, (15, 15)).flatten()
predicted_label = model.predict([test_image])
print(f'Predicted label for test image: {catelog[predicted_label[0]]}')
print(f'Predicted label index: {predicted_label[0]}')