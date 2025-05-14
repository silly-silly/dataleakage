

def __phi__(phi_0, phi_1):
    if phi_0:
        return phi_0
    return phi_1

def set_field_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def set_index_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def global_wrapper(x):
    return x
import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import LinearRegression, train_test_split
_var0 = 'data.csv'
inputs = pd.read_csv(_var0)
_var1 = 'label'
y = inputs[_var1]
_var2 = 'label'
data0 = inputs.drop(_var2)
data = data0
(_var3, _var4, _var5, _var6) = train_test_split(data, y)
X_train_0 = _var3
y_train = _var4
X_test_0 = _var5
y_test = _var6
_var7 = 50
select = SelectPercentile(chi2, percentile=_var7)
select_0 = select.fit(X_train_0)
X_train = select_0.transform(X_train_0)
X_test = select_0.transform(X_test_0)
model = LinearRegression()
model_0 = model.fit(X_train, y_train)
model_0.score(X_test, y_test)
