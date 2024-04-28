"""Based on the report result the classifier with entrophy criterion bette predicts the result (it has better accuracy, precision, recall) than classifier with ginni criterion"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
# import graphviz
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("suv.csv")
print(df.head())

X = df[['Age', 'EstimatedSalary']]
y = df[['Purchased']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=11)

X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

# entropy
classifier = DecisionTreeClassifier(criterion="entropy",max_depth=4)
classifier.fit(X_train_std, y_train)
y_pred = classifier.predict(X_test_std)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  gini index
classifier = DecisionTreeClassifier()
classifier.fit(X_train_std, y_train)
y_pred = classifier.predict(X_test_std)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
