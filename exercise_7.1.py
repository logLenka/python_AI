"""we have received better result using radial kernel than linear kernel (0.99 vs 1.00 accuracy)"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df2 = pd.read_csv("data_banknote_authentication.csv")
print(df2.head())

X = df2.drop('class', axis=1)
y = df2['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



# radial
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))