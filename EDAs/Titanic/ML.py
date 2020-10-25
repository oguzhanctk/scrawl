#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
#%%

#import csv files
df = pd.read_csv("original.csv", sep=";")
df_poly = pd.read_csv("PLR.csv", sep=";")
X = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)
z = np.array([1, 5, 10, 15, 20, 25]).reshape(-1, 1)
test_1 = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
test_2 = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
#%%

linear_reg_model = LinearRegression()
linear_reg_model.fit(test_1, test_2)
preds = linear_reg_model.predict(test_1)
print(linear_reg_model.score(test_1, test_2), r2_score(preds, test_2))


# %%

#visualize undependent variables
f, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(20, 10))
sns.regplot(df.deneyim, df.maas, fit_reg=True, ax=axes[0])
sns.regplot(df.yas , df.maas, fit_reg=True, ax=axes[1])
plt.show()

#define inputs and outputs

X_1 = df.iloc[:, [0,1]]
y_1 = df.maas

MLR_model = LinearRegression()

MLR_model.fit(X_1, y_1) 

preds_1 = MLR_model.predict([[21, 21], [21, 1], [25, 5]])

# %%

#polynomial linear regression -> outputun bir limite yaklaştığı durumlar

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

y = df_poly.iloc[:, 1].values.reshape(-1, 1)
X_fiyat = df_poly.iloc[:, 0].values.reshape(-1, 1)

make_polynom = PolynomialFeatures(degree=4)
PLR_model = LinearRegression()

poly_X_fiyat = make_polynom.fit_transform(X_fiyat)

PLR_model.fit(poly_X_fiyat, y)

preds_2 = PLR_model.predict(poly_X_fiyat)

err = mean_squared_error(preds_2, y)
print(err)
plt.scatter(df_poly.araba_fiyat, df_poly.araba_max_hiz)
plt.plot(df_poly.araba_fiyat, preds_2)

# %%
#Decision tree regressor

mutated_X = np.arange(min(X_fiyat), max(X_fiyat), 0.1).reshape(-1, 1)

DTR_model = DecisionTreeRegressor()
DTR_model.fit(X_fiyat, y)
preds_3 = DTR_model.predict(mutated_X)

# err_for_DTR = mean_squared_error(preds_3, y)

plt.scatter(X_fiyat, y)
plt.plot(mutated_X, preds_3)
# %%

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

RFR = RandomForestRegressor(n_estimators=100, random_state=23)
RFR.fit(poly_X_fiyat, y)
preds_4 = RFR.predict(poly_X_fiyat)
# print(r2_score(preds_4, y))

#visualize observed and predicted values
plt.plot(X_fiyat, y)
plt.plot(X_fiyat, preds_4)
plt.show()

# %%


#scrawl about linear regression

df_test_linear = pd.read_csv("linear_test.csv")
X_sat = df_test_linear.iloc[:, 0].values.reshape(-1, 1)
y_gpa = df_test_linear.iloc[:, 1].values.reshape(-1, 1)

make_sat_polynom = PolynomialFeatures(degree=4)
X_poly_sat = make_sat_polynom.fit_transform(X_sat)
X_train, X_test, y_train, y_test = train_test_split(X_poly_sat, y_gpa, test_size=0.2, random_state=23)

#visualize X_input and y_output

test_reg_model = LinearRegression()
test_reg_model.fit(X_train, y_train)
test_preds = test_reg_model.predict(X_test)
print(r2_score(y_test, test_preds))

#visualize

plt.scatter(np.arange(0, 17), test_preds)

# %%

#Logistic Regression -> (classification)    

arr = [1, 10, 100]

df_arr = pd.DataFrame({"column_1" : arr})






# %%
