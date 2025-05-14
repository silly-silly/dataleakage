

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
import numpy as np
import pandas as pd
_var0 = '../input/train.csv'
train = pd.read_csv(_var0)
_var1 = '../input/test.csv'
test = pd.read_csv(_var1)
train.head()
test.head()
_var2 = [train, test]
_var3 = False
data = pd.concat(_var2, sort=_var3)
data.head()
_var4 = len(train)
_var5 = len(test)
_var6 = len(data)
_var7 = (_var4, _var5, _var6)
print(_var7)
_var8 = data.isnull()
_var8.sum()
_var9 = 'Sex'
_var10 = data[_var9]
_var11 = ['male', 'female']
_var12 = [0, 1]
_var13 = True
_var10.replace(_var11, _var12, inplace=_var13)
_var14 = 'Embarked'
_var15 = data[_var14]
_var16 = 'S'
_var17 = True
_var15.fillna(_var16, inplace=_var17)
_var18 = 'Embarked'
_var19 = 'Embarked'
_var20 = data[_var19]
_var21 = {'S': 0, 'C': 1, 'Q': 2}
_var22 = _var20.map(_var21)
_var23 = _var22.astype(int)
data_0 = set_index_wrapper(data, _var18, _var23)
_var24 = 'Fare'
_var25 = data_0[_var24]
_var26 = 'Fare'
_var27 = data_0[_var26]
_var28 = np.mean(_var27)
_var29 = True
_var25.fillna(_var28, inplace=_var29)
_var30 = 'Age'
_var31 = data_0[_var30]
age_avg = _var31.mean()
_var32 = 'Age'
_var33 = data_0[_var32]
age_std = _var33.std()
_var34 = 'Age'
_var35 = data_0[_var34]
_var36 = np.random
_var37 = (age_avg - age_std)
_var38 = (age_avg + age_std)
_var39 = _var36.randint(_var37, _var38)
_var40 = True
_var35.fillna(_var39, inplace=_var40)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
_var41 = 1
_var42 = True
data_0.drop(delete_columns, axis=_var41, inplace=_var42)
_var43 = len(train)
train_0 = data_0[:_var43]
_var44 = len(train_0)
test_0 = data_0[_var44:]
_var45 = 'Survived'
y_train = train_0[_var45]
_var46 = 'Survived'
_var47 = 1
X_train = train_0.drop(_var46, axis=_var47)
_var48 = 'Survived'
_var49 = 1
X_test = test_0.drop(_var48, axis=_var49)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
_var50 = 'l2'
_var51 = 'sag'
_var52 = 0
clf = LogisticRegression(penalty=_var50, solver=_var51, random_state=_var52)
clf_0 = clf.fit(X_train, y_train)
y_pred = clf_0.predict(X_test)
