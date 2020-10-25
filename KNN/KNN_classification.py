#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

#%%
#import data define X and y
cancer_df = pd.read_csv("cancer.csv")
X = cancer_df.drop(["Unnamed: 32", "id", "diagnosis"], axis=1)
y = [1 if x == "M" else 0 for x in cancer_df.diagnosis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))

#visualization
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.scatterplot(X.perimeter_mean, X.compactness_mean, hue=cancer_df.diagnosis, ax=ax[0])
sns.scatterplot(X.area_mean, X.perimeter_mean, hue=cancer_df.diagnosis, ax=ax[1])
plt.show()

# %%
#find best hyperporameter
score_list = []
for _i in range(1, 20):
    knn_c = KNeighborsClassifier(n_neighbors=_i)
    knn_c.fit(X_train, y_train)
    score_list.append(knn_c.score(X_test, y_test))
    print("accuracy score for {} is {}".format(_i, knn_c.score(X_test, y_test)))

plt.plot(range(1, 20), score_list) #we find best n_neigbours param is 8

#%%
#define model and test data
knn_classifier = KNeighborsClassifier(n_neighbors=8)
knn_classifier.fit(X_train, y_train)
print("accuracy score : {}".format(knn_classifier.score(X_test, y_test)))


# %%
