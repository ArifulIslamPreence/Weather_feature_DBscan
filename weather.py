import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler

pdf = pd.read_csv("weather-stations20140101-20141231.csv")
pdf.head(5)

pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)

rcParams['figure.figsize'] = (14,10)

llon = -140
ulon = -50
llat = 40
ulat = 65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]

myMap = Basemap(projection = 'merc',
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon = llon, llcrnrlat = llat,
                urcrnrlon = ulon, urclrnrlat = ulat)

myMap.drawcoastlines()
myMap.drawcountries()
myMap.fillcontinents(color = 'white', alpha = 0.3)
myMap.shadedrelief()

xs,ys = myMap(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

for index,row in pdf.iterrows():
    myMap.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)

plt.show()

sklearn.utils.check_random_state(1000)
clus_dataset = pdf[['xm','ym']]
clus_dataset = np.nan_to_num(clus_dataset)
clus_dataset = StandardScaler().fit_transform(clus_dataset)

db = DBSCAN(eps=0.5,min_samples=10).fit_transform(clus_dataset)
core_sample_mask = np.zeros_like(db.labels_, dtype=bool)

core_sample_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["clus_db"] = labels

realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

pdf[["Stn_name","Tx","Tm","clus_db"]].head(5)

colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]
    myMap.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

