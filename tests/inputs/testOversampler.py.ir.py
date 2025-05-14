

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
from sklearn.datasets import fetch_openml
from imblearn.datasets import make_imbalance
_var2 = 1119
_var3 = True
_var4 = True
(_var0, _var1) = fetch_openml(data_id=_var2, as_frame=_var3, return_X_y=_var4)
X = _var0
y = _var1
_var5 = 'number'
X_0 = X.select_dtypes(include=_var5)
_var8 = {'>50K': 300}
_var9 = 1
(_var6, _var7) = make_imbalance(X_0, y, sampling_strategy=_var8, random_state=_var9)
X_1 = _var6
y_0 = _var7
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
_var10 = 0
sampler = RandomOverSampler(random_state=_var10)
(_var11, _var12) = sampler.fit_resample(X_1, y_0)
X_resampled = _var11
y_resampled = _var12
_var17 = 0.2
_var18 = 0
(_var13, _var14, _var15, _var16) = train_test_split(X_resampled, y_resampled, test_size=_var17, random_state=_var18)
X_train = _var13
X_test = _var14
y_train = _var15
y_test = _var16
_var19 = 0
model = HistGradientBoostingClassifier(random_state=_var19)
model_0 = model.fit(X_train, y_train)
model_0.predict(X_test)
