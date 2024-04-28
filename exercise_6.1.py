"""
based on the result from the heat map there doesn't seem to be a srtomg correlation between the respective variables.
LogisticRegression vs. k-nearest neighbors metrics results:
    Accuracy (the probability of correctly predicted result, ): 0.89 vs 0.86
    Precision (how often the system predicts correctly): 0.89 vs. 0.89
    Recall: 0.997 vs. 0.96 
Our model is much better at preddicting 1s (y_no) than 0s as there is much more 'no' answers for the target variable (y - has the client subscribed a term deposit?)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('bank.csv', sep=";")
df2 = df.loc[:, ['y', 'job','marital','default','housing','poutcome']]
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])
# print(df3.head())
# print(df2.head())
sns.heatmap(data=df3.corr().round(2), annot=True)
plt.show()


X = df3.iloc[:, 1:]
# y = df3['y']
y = pd.get_dummies(df3,columns=['y']).iloc[:, -2]
# df4 = pd.get_dummies(df3,columns=['y'])
# print(df4.head())


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# print(X_train.shape)

# LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# metrics.plot_confusion_matrix(model, X_test, y_test)

metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix).plot()
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

# k-nearest neighbors
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


print(classification_report(y_test, y_pred))






