from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 10
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

acc = clf.score(X_test, y_test)
print(acc)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test),digits=4))

with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n\n")
        outfile.write(str(confusion_matrix(y_test, clf.predict(X_test))) + "\n\n")
        outfile.write(str(classification_report(y_test, clf.predict(X_test),digits=4)) + "\n")   




# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

