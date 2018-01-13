import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.cluster import KMeans
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dogDataset = pd.read_csv('AKCBreedInfo.csv',',')

for i in dogDataset['weight_high_lbs']:
    X = dogDataset['weight_high_lbs'] / 2.2 

for i in dogDataset['height_high_inches']:
    # Ya = dogDataset['height_high_inches'] * 2.54
    Y = dogDataset['height_high_inches'] * 2.54


for i in dogDataset['Breed']:
    name = dogDataset['Breed']

data = np.array(list(zip(X, Y)))
data  = pd.DataFrame(data)
data.columns = ['peso', 'altura']

model = KMeans(n_clusters=3)
model.fit(data)

y = model.labels_

vetor = np.array([data.peso,data.altura])
#print(vetor)
class_names = ['pequeno', 'medio', 'grande']

 # Split the data into a training set and a test set
data_train, data_test, y_train, y_test = train_test_split(data, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(data_train, y_train).predict(data_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    print('Specificity : ', specificity1)
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

