#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

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

#%%
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print("accuract score : {}".format(nb_model.score(X_test, y_test)))
print("classification report :\n {}".format(classification_report(y_test, preds)))
# %%

# %%

