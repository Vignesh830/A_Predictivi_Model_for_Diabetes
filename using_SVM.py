import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #(standardize data to common range)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 # Loading the diabetes dataset to pandas dataframe
df=pd.read_csv("/content/sample_data/diabetes.csv")
df["Outcome"].value_counts()
df.shape
df.groupby("Outcome").mean()
X= df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y, random_state=2)
print(X.shape,X_train.shape,X_test.shape) 
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

#accuracy of testing data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy_score=accuracy_score(y_train,X_train_prediction)
print(f"Accuracy Score of training data : {training_data_accuracy_score * 100} %")

# accuracy_score of testing data

X_test_prediction=classifier.predict(X_test)
testing_data_accuracy_score=accuracy_score(y_test,X_test_prediction)
print(f"Accuracy Score of testing data : {testing_data_accuracy_score * 100} %")

##graph stuff
# Plot the distribution of the target variable
plt.hist(y, bins=10, edgecolor="white")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.title("Distribution of Outcome in Diabetes Dataset")
plt.show()

# Plot the correlation matrix of the features
corr_matrix = X.corr()
plt.matshow(corr_matrix, cmap="coolwarm")
plt.colorbar()
plt.title("Correlation Matrix of Diabetes Features")
plt.show()

# Plot the mean of the features by outcome
plt.figure(figsize=(10, 6))
df.groupby("Outcome").mean().plot(kind="bar")
plt.xlabel("Outcome")
plt.ylabel("Mean Feature Value")
plt.title("Mean Feature Value by Outcome in Diabetes Dataset")
plt.show()