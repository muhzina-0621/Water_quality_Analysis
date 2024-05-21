import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


data = pd.read_csv('water.csv')

# print(data.shape)
data["ph"].fillna(value = data["ph"].mean(), inplace = True)
data["Sulfate"].fillna(value = data["Sulfate"].mean(), inplace = True)
data["Trihalomethanes"].fillna(value = data["Trihalomethanes"].mean(), inplace = True)

x=data.drop('Potability',axis=1)
data=data.drop("ph",axis=1)
data=data.drop("Hardness",axis=1)
data=data.drop("Solids",axis=1)
data=data.drop("Chloramines",axis=1)
data=data.drop("Sulfate",axis=1)
data=data.drop("Conductivity",axis=1)
data=data.drop("Organic_carbon",axis=1)
data=data.drop("Trihalomethanes",axis=1)
data=data.drop("Turbidity",axis=1)
x = pd.DataFrame(x.values)
# print(x)
y=data['Potability']
# print(y)

# print(X_train,y_train)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
model=svclassifier.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
cm=metrics.confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
plt.show()
print(classification_report(y_test,y_pred))
print("SVM score: ",svclassifier.score(X_test, y_test))
Accuracy = metrics.accuracy_score(y_test, y_pred)
print(Accuracy)