# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 02:22:46 2017
    
@author: SainiD
"""
# Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Importing the dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

"""X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index1 = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index1], y_train[shuffle_index1]
shuffle_index2=np.random.permutation(10000)
X_test, y_test = X_test[shuffle_index2], y_test[shuffle_index2]"""

# Fitting the KNeighborsClassifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

#Grabing an instance’s feature vector, reshaping and dipaying it
digit = X_test[20]
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

#Predicting the digit using the K-NN Classifier
predict = knn_clf.predict([digit])
print('prediction is',predict)
print('label is',y_test[20])

# Predicting the Test Set Results
y_predict=knn_clf.predict(X_test)
print('\nPredicted Labels for Test Images: ',y_predict)

#Performing K-fold cross-validation on the Classifier
from sklearn.model_selection import cross_val_score
crs_vld=cross_val_score(knn_clf, X_train, y_train, cv=3, scoring="accuracy")
print('\nThe Cross-Validation score is', crs_vld)

#Generating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print('\nThe Confusion Matrix for Test Data: \n',cm)

# Plot Confusion Matrix for Test Data
plt.matshow(cm)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show the Test Images with Original and Predicted Labels
a = np.random.randint(1,500,10)
for i in a:
	two_d = (np.reshape(X_test[i], (28, 28)) * 255).astype(np.uint8)
	plt.title('Original Label: {0}  Predicted Label: {1}'.format(y_test[i],y_predict[i]))
	plt.imshow(two_d, interpolation='nearest',cmap='gray')
	plt.show()

#Calculating Accuracy Score of the Classifier Model
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_predict)
print('\nThe Models Accuracy is', accuracy)

#Calculating the Mean Squared Error(MSE)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_predict)
print('\nThe test error for the Classifier is',mse)