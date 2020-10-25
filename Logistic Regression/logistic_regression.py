#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from itertools import product

# %%
#functions
def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss

def log_reg_optimize_parameter(param_dict, X, y): #return accuracy score with different parameters
    ranges = []
    keys = []
    log_reg = LogisticRegression()
    for key, value in param_dict.items():
        ranges.append(value)
        keys.append(key)
    try:
        for x in list(product(*ranges)):
            d = {}
            for _i in range(len(x)):
                d.update({keys[_i] : x[_i]})
            log_reg.set_params(**d)
            log_reg.fit(X, y)
            print("Score : {} for parameters {}".format(log_reg.score(X, y), d))
    except ValueError:
        print("Enter valid parameter")
    return ranges, keys
    
#import data and define y and X
cancer_df = pd.read_csv("cancer.csv")
y = [1 if x == "M" else 0 for x in cancer_df.diagnosis]
X = cancer_df.drop(["Unnamed: 32", "id" ,"diagnosis"], axis=1)

#normalize dataframe values between 0 and 1
X_normalized = (X - np.min(X)) / (np.max(X) - np.min(X))

#split data for train and test and get transpose
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
X_train = X_train.T
X_test = X_test.T

params = {
    "max_iter" : range(1, 11),
    "intercept_scaling" : [0.1, 0.5, 1]
}

#find best parameter
log_reg_optimize_parameter(params, X_train.T, y_train)

#from sklearn.linear_model import SGDClassifier #stockhastic gradient descent classifier

# define model
log_reg_model = LogisticRegression(max_iter=100, random_state=42, verbose=1)
log_reg_model.fit(X_train.T, y_train)
lr_preds = log_reg_model.predict(X_test.T)
print("accuracy : {}".format(log_reg_model.score(X_test.T, y_test)))








# %%
