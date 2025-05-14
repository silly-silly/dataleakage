

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
import numpy as np
_var0 = 'data.csv'
df = pd.read_csv(_var0)
l = len(df)
_var1 = 'Survived'
y = df[_var1]
_var2 = 'Survived'
_var3 = 1
df_0 = df.drop(_var2, axis=_var3)
_var4 = 'Fare'
_var5 = 'Fare'
_var6 = df_0[_var5]
_var7 = 'Fare'
_var8 = df_0[_var7]
_var9 = np.mean(_var8)
_var10 = False
_var11 = _var6.fillna(_var9, inplace=_var10)
df_1 = set_index_wrapper(df_0, _var4, _var11)
from sklearn.model_selection import train_test_split
_var16 = 0.33
_var17 = 42
(_var12, _var13, _var14, _var15) = train_test_split(df_1, y, test_size=_var16, random_state=_var17)
X_train = _var12
X_test = _var13
y_train = _var14
y_test = _var15
from sklearn.linear_model import LogisticRegression
_var18 = 'l2'
_var19 = 'sag'
_var20 = 0
clf = LogisticRegression(penalty=_var18, solver=_var19, random_state=_var20)
clf_0 = clf.fit(X_train, y_train)
y_pred = clf_0.predict(X_test)
