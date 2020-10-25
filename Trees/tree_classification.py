#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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
#define model and find best parameters

best_params = {'min_samples_split': 3, 'min_samples_leaf': 2, 'max_depth': 9}
dec_tree_classifier = DecisionTreeClassifier(**best_params, random_state=23)
# params = {
#     "max_depth" : np.arange(5, 10),
#     "min_samples_leaf" : np.arange(1, 5),
#     "min_samples_split" : np.arange(1, 5)
# }
# grid_search = RandomizedSearchCV(
#     estimator=dec_tree_classifier,
#     param_distributions=params,
#     n_iter=50,
#     n_jobs=-1,
#     verbose=10
# )

# grid_search.fit(X_train, y_train) #{'min_samples_split': 3, 'min_samples_leaf': 2, 'max_depth': 9}
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# print(grid_search.best_params_)

dec_tree_classifier.fit(X_train, y_train)
print("accuracy_score : {}".format(dec_tree_classifier.score(X_test, y_test)))

# %%
#define model and find best parameters
rand_for_classifier = RandomForestClassifier(random_state=23, n_jobs=-1, max_depth=7)
# grid_search_rf = RandomizedSearchCV( #found max_depth=7 is the best parameter
#     estimator=rand_for_classifier,
#     param_distributions=params,
#     n_jobs=-1,
#     n_iter=50,
#     verbose=10
# )
# grid_search_rf.fit(X_test, y_test)
# print(grid_search_rf.best_params_, grid_search_rf.best_score_)

rand_for_classifier.fit(X_train, y_train)
print("accuracy score : {}".format(rand_for_classifier.score(X_test, y_test)))
y_preds = rand_for_classifier.predict(X_test)


# %%
