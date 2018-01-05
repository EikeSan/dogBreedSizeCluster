import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import stats
from sklearn.cluster import KMeans
from math import sqrt
import random
import seaborn as sns

dogDataset = pd.read_csv('AKCBreedInfo.csv',',')

#Peso em Kilos e altura em centimetros
#print(dogDataset['Breed'])
for i in dogDataset['weight_high_lbs']:
    X = dogDataset['weight_high_lbs'] / 2.2 
    # Xa = dogDataset['weight_high_lbs'] / 2.2 
    #X = round(X,2)

# for i in dogDataset['weight_low_lbs']:
#     Xb = dogDataset['weight_low_lbs'] / 2.2


for i in dogDataset['height_high_inches']:
    # Ya = dogDataset['height_high_inches'] * 2.54
    Y = dogDataset['height_high_inches'] * 2.54

# for i in dogDataset['height_low_inches']:
#     Yb = dogDataset['height_low_inches'] * 2.54

for i in dogDataset['Breed']:
    name = dogDataset['Breed']

# X = Xa - Xb
# Y = Ya - Yb
# print(X)
# print(Y)

data = np.array(list(zip(X, Y)))
data  = pd.DataFrame(data)
data.columns = ['peso', 'altura']

model = KMeans(n_clusters=3)
model.fit(data)
print data.peso[0]
print data.altura[0]
print(model.predict(data.peso[0]git ))


#print(model.labels_)
#labels = np.array([model.labels_])
#colors = np.array(['r', 'g', 'b'])
#LABEL_COLOR_MAP = {0 : 'r', 1 : 'g', 2 : 'b'}
#labelColor = [LABEL_COLOR_MAP[l] for l in labels]

data = np.array(list(zip(X, Y, model.labels_)))
data  = pd.DataFrame(data)
data.columns = ['peso', 'altura','clusters']

#print(model.cluster_centers_)
#print(model.cluster_centers_[0][0])

#centroids = np.array(list(zip(model.cluster_centers_)))
#centroid = pd.DataFrame(centroids)
#centroid.columns = ['x','y']

#print(centroid.x[0])

#plt.legend() 
# for centroid in model.cluster_centers_:

# plt.scatter(model.cluster_centers_[0][0],model.cluster_centers_[0][1],color=colors[0], s = 60, marker = "o", label= 'Pequeno')
# plt.scatter(model.cluster_centers_[1][0],model.cluster_centers_[1][1],color=colors[1], s = 60, marker = "o", label= 'Medio') 
# plt.scatter(model.cluster_centers_[2][0],model.cluster_centers_[2][1],color=colors[2], s = 60, marker = "o", label =  'Grande')

# plt.scatter(data.peso, data.altura, c=colors[model.labels_], s=60)

sns.lmplot('peso', 'altura', data=data, fit_reg=False, hue='clusters',  scatter_kws={"marker": "D", "s": 60},size = 5, aspect = 2)
plt.title('Cluster Dog Breed Size')
plt.ylabel('Altura em Centimetros')
plt.xlabel('Peso em Kilos')
#plt.legend()
plt.savefig('clusterDog.png')
#plt.show()