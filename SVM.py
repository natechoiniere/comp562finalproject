import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

hog_matrix = np.load("hog_matrix_small.npy")

with open("labels_small.pkl", "rb") as f:
    label_list = pickle.load(f)

#Encode the labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(label_list)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_matrix, encoded_labels, test_size=0.2, random_state=2)

#SVM (defaults to gaussian)
svm_classifier = SVC(C=0.01, verbose=1)

#Train the classifier
svm_classifier.fit(X_train, y_train)

#Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

#Evaluate the classifier
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")
