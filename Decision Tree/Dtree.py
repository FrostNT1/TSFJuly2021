# Importing Libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Importing Dataset
df = load_iris()
X = pd.DataFrame(df.data, columns=df.feature_names)
y = df.target

# Spliting traing and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))


from sklearn.tree import plot_tree
plot_tree(classifier, feature_names=df.feature_names,
          class_names=['setosa', 'versicolor', 'virginica'],
          filled=True)
print()