

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
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
import matplotlib.pyplot as plt
_var3 = '../data/wines.csv'
df = pd.read_csv(_var3)
df.head()
_var4 = 'Class'
y = df[_var4]
y.value_counts()
y_cat = pd.get_dummies(y)
y_cat.head()
_var5 = 'Class'
_var6 = 1
X = df.drop(_var5, axis=_var6)
X.shape
import seaborn as sns
_var7 = 'Class'
sns.pairplot(df, hue=_var7)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xsc = sc.fit_transform(X)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K
K.clear_session()
model = Sequential()
_var8 = 5
_var9 = 13
_var10 = (_var9,)
_var11 = 'he_normal'
_var12 = 'relu'
_var13 = Dense(_var8, input_shape=_var10, kernel_initializer=_var11, activation=_var12)
model.add(_var13)
_var14 = 3
_var15 = 'softmax'
_var16 = Dense(_var14, activation=_var15)
model.add(_var16)
_var17 = 0.1
_var18 = RMSprop(lr=_var17)
_var19 = 'categorical_crossentropy'
_var20 = ['accuracy']
model.compile(_var18, _var19, metrics=_var20)
_var21 = y_cat.values
_var22 = 8
_var23 = 10
_var24 = 1
_var25 = 0.2
model_0 = model.fit(Xsc, _var21, batch_size=_var22, epochs=_var23, verbose=_var24, validation_split=_var25)
K.clear_session()
model_1 = Sequential()
_var26 = 8
_var27 = 13
_var28 = (_var27,)
_var29 = 'he_normal'
_var30 = 'tanh'
_var31 = Dense(_var26, input_shape=_var28, kernel_initializer=_var29, activation=_var30)
model_1.add(_var31)
_var32 = 5
_var33 = 'he_normal'
_var34 = 'tanh'
_var35 = Dense(_var32, kernel_initializer=_var33, activation=_var34)
model_1.add(_var35)
_var36 = 2
_var37 = 'he_normal'
_var38 = 'tanh'
_var39 = Dense(_var36, kernel_initializer=_var37, activation=_var38)
model_1.add(_var39)
_var40 = 3
_var41 = 'softmax'
_var42 = Dense(_var40, activation=_var41)
model_1.add(_var42)
_var43 = 0.05
_var44 = RMSprop(lr=_var43)
_var45 = 'categorical_crossentropy'
_var46 = ['accuracy']
model_1.compile(_var44, _var45, metrics=_var46)
_var47 = y_cat.values
_var48 = 16
_var49 = 20
_var50 = 1
model_2 = model_1.fit(Xsc, _var47, batch_size=_var48, epochs=_var49, verbose=_var50)
model_2.summary()
_var51 = model_2.layers
_var52 = 0
_var53 = _var51[_var52]
inp = _var53.input
_var54 = model_2.layers
_var55 = 2
_var56 = _var54[_var55]
out = _var56.output
_var57 = [inp]
_var58 = [out]
features_function = K.function(_var57, _var58)
_var59 = [Xsc]
_var60 = features_function(_var59)
_var61 = 0
features = _var60[_var61]
features.shape
_var62 = 0
_var63 = features[:, _var62]
_var64 = 1
_var65 = features[:, _var64]
plt.scatter(_var63, _var65, c=y_cat)
from keras.layers import Input
from keras.models import Model
K.clear_session()
_var66 = 13
_var67 = (_var66,)
inputs = Input(shape=_var67)
_var68 = 8
_var69 = 'he_normal'
_var70 = 'tanh'
x = Dense(_var68, kernel_initializer=_var69, activation=_var70)(inputs)
_var71 = 5
_var72 = 'he_normal'
_var73 = 'tanh'
x_0 = Dense(_var71, kernel_initializer=_var72, activation=_var73)(x)
_var74 = 2
_var75 = 'he_normal'
_var76 = 'tanh'
second_to_last = Dense(_var74, kernel_initializer=_var75, activation=_var76)(x_0)
_var77 = 3
_var78 = 'softmax'
outputs = Dense(_var77, activation=_var78)(second_to_last)
model_3 = Model(inputs=inputs, outputs=outputs)
_var79 = 0.05
_var80 = RMSprop(lr=_var79)
_var81 = 'categorical_crossentropy'
_var82 = ['accuracy']
model_3.compile(_var80, _var81, metrics=_var82)
_var83 = y_cat.values
_var84 = 16
_var85 = 20
_var86 = 1
model_4 = model_3.fit(Xsc, _var83, batch_size=_var84, epochs=_var85, verbose=_var86)
_var87 = [inputs]
_var88 = [second_to_last]
features_function_0 = K.function(_var87, _var88)
_var89 = [Xsc]
_var90 = features_function_0(_var89)
_var91 = 0
features_0 = _var90[_var91]
_var92 = 0
_var93 = features_0[:, _var92]
_var94 = 1
_var95 = features_0[:, _var94]
plt.scatter(_var93, _var95, c=y_cat)
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
_var96 = '/tmp/udemy/weights.hdf5'
_var97 = 1
_var98 = True
checkpointer = ModelCheckpoint(filepath=_var96, verbose=_var97, save_best_only=_var98)
_var99 = 'val_loss'
_var100 = 0
_var101 = 1
_var102 = 1
_var103 = 'auto'
earlystopper = EarlyStopping(monitor=_var99, min_delta=_var100, patience=_var101, verbose=_var102, mode=_var103)
_var104 = '/tmp/udemy/tensorboard/'
tensorboard = TensorBoard(log_dir=_var104)
from sklearn.model_selection import train_test_split
_var109 = y_cat.values
_var110 = 0.3
_var111 = 42
(_var105, _var106, _var107, _var108) = train_test_split(Xsc, _var109, test_size=_var110, random_state=_var111)
X_train = _var105
X_test = _var106
y_train = _var107
y_test = _var108
K.clear_session()
_var112 = 13
_var113 = (_var112,)
inputs_0 = Input(shape=_var113)
_var114 = 8
_var115 = 'he_normal'
_var116 = 'tanh'
x_1 = Dense(_var114, kernel_initializer=_var115, activation=_var116)(inputs_0)
_var117 = 5
_var118 = 'he_normal'
_var119 = 'tanh'
x_2 = Dense(_var117, kernel_initializer=_var118, activation=_var119)(x_1)
_var120 = 2
_var121 = 'he_normal'
_var122 = 'tanh'
second_to_last_0 = Dense(_var120, kernel_initializer=_var121, activation=_var122)(x_2)
_var123 = 3
_var124 = 'softmax'
outputs_0 = Dense(_var123, activation=_var124)(second_to_last_0)
model_5 = Model(inputs=inputs_0, outputs=outputs_0)
_var125 = 0.05
_var126 = RMSprop(lr=_var125)
_var127 = 'categorical_crossentropy'
_var128 = ['accuracy']
model_5.compile(_var126, _var127, metrics=_var128)
_var129 = 32
_var130 = 20
_var131 = 2
_var132 = (X_test, y_test)
_var133 = [checkpointer, earlystopper, tensorboard]
model_6 = model_5.fit(X_train, y_train, batch_size=_var129, epochs=_var130, verbose=_var131, validation_data=_var132, callbacks=_var133)
model_6.evaluate(X_test, y_test)
