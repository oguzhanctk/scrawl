#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as time
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

#%%
#K-Means clustering
#creating data
x1 = np.random.normal(35, 5, 100)
y1 = np.random.normal(35, 5, 100)
x2 = np.random.normal(65, 5, 100)
y2 = np.random.normal(65, 5, 100)
x3 = np.random.normal(10, 5, 100)
y3 = np.random.normal(10, 5, 100)

data_x = np.concatenate((x1, x2, x3), axis=0)
data_y = np.concatenate((y1, y2, y3), axis=0)

df = pd.DataFrame({"x" : data_x, "y" : data_y})

#given data visualization
plt.scatter(df.x, df.y)
# %%
#find optimal k value
wcss = []
for _i in range(1, 12):
    test_kmeans = KMeans(n_clusters=_i)
    test_kmeans.fit(df)
    wcss.append(test_kmeans.inertia_)

# plt.plot(range(1, 12), wcss) #k = 3 (from elbow rule, it is clear that k=3)

#create model to define clusters within data
k_means = KMeans(n_clusters=3)
k_means.fit(df)
df["label"] = k_means.labels_
sns.scatterplot(df.x, df.y, hue="label", data=df)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1])

# %%
#Hierarchical clustering

#find optimal cluster number
t = time.time()
merg = linkage(df, "ward")
dendrogram(merg, leaf_rotation=90) #looks like 3 is optimal
plt.show()
elapsed = time.time() - t
print("elapsed time : {}".format(elapsed))

# %%
#hierarchical clustering

hier_clus = AgglomerativeClustering(n_clusters=3)
df["label"] = hier_clus.fit_predict(df)

#visualization
sns.scatterplot(data=df, x="x", y="y", hue="label", palette="bright")

# %%
#k-means images example
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np

china = load_sample_image("china.jpg")
data = china / 255.0
data = data.reshape(427 * 640, 3)
# %%
from sklearn.cluster import MiniBatchKMeans

#cluster millions color to just 16 cluster(color)
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
# %%
#convert all colors to new 16 cluster_center colors
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
china_recolored = new_colors.reshape(china.shape)
# %%
#visualization 
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))

ax[0].imshow(china)
ax[0].set_title("original image", size=16)

ax[1].imshow(china_recolored)
ax[1].set_title("16-color image", size=16)

plt.show()

# %%
