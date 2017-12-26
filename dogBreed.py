import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import stats
from sklearn.cluster import KMeans
from math import sqrt
import random

dogDataset = pd.read_csv('AKCBreedInfo.csv',',')

#Peso em Kilos e altura em centimetros
#print(dogDataset['Breed'])
for i in dogDataset['weight_high_lbs']:
    X = dogDataset['weight_high_lbs'] / 2.2
    X = round(X,2)

for i in dogDataset['height_high_inches']:
    Y = dogDataset['height_high_inches'] * 2.54

for i in dogDataset['Breed']:
    name = dogDataset['Breed']


data = np.array(list(zip(X, Y)))
data  = pd.DataFrame(data)
data.columns = ['peso', 'altura']

# < 168 - min
# 525 - pqn
# 1225 - me 
# 3105 - gr
# > 3105 - gigante
#Primeiro definimos a função para calculo de distância.
model = KMeans(n_clusters=3)
model.fit(data)

#print(model.labels_)

colors = np.array(['r', 'g', 'b', 'y', 'm'])

plt.ylabel('Altura em Centimetros')
plt.xlabel('Peso em Kilos')
plt.legend()

plt.scatter(data.peso, data.altura, c=colors[model.labels_], s=60)
plt.title('Cluster Dog Breed Size')
plt.savefig('clusterDog.png')
plt.show()