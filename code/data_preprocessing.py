import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Reading dataset and storing
dataset = pd.read_csv('ds1.csv',header=None)
dataset.replace('?', np.nan, inplace=True)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,279].values


#deleting column with maximum values missing
dataset.head()
X=np.delete(X,13,1)


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,0:279])
X[:,0:279]=imputer.transform(X[:,0:279])


for i in range(0,452):
	if (y[i]>=14):
		y[i]=y[i]-3
        
#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

#Plotting a histogram
x=np.arange(1,17)
h,bins=np.histogram(y,16)
plt.bar(x-0.4,h)
plt.xlabel('Class Labels')
plt.ylabel('Number of Instances')
plt.xticks(x)

np.savetxt("reduced_features_X1.csv",X, fmt='%s', delimiter=",")
np.savetxt("feature_y1.csv",y, fmt='%s', delimiter=",")

