

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
import matplotlib.pyplot as plt
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
_var3 = plt.rcParams
_var4 = 'figure.dpi'
_var5 = 200
_var3_0 = set_index_wrapper(_var3, _var4, _var5)
_var6 = 3
_var7 = True
np.set_printoptions(precision=_var6, suppress=_var7)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
import tensorflow as tf
import numpy as np
_var8 = np.random
_var9 = 100
_var10 = _var8.rand(_var9)
_var11 = np.float32
x_data = _var10.astype(_var11)
_var12 = 0.1
_var13 = (x_data * _var12)
_var14 = 0.3
y_data = (_var13 + _var14)
_var15 = [1]
_var16 = (- 1.0)
_var17 = 1.0
_var18 = tf.random_uniform(_var15, _var16, _var17)
W = tf.Variable(_var18)
_var19 = [1]
_var20 = tf.zeros(_var19)
b = tf.Variable(_var20)
_var21 = (W * x_data)
y = (_var21 + b)
_var22 = (y - y_data)
_var23 = tf.square(_var22)
loss = tf.reduce_mean(_var23)
_var24 = tf.train
_var25 = 0.5
optimizer = _var24.GradientDescentOptimizer(_var25)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
_var26 = 201
_var27 = range(_var26)
for step in _var27:
    sess.run(train)
    _var28 = 20
    _var29 = (step % _var28)
    _var30 = 0
    _var31 = (_var29 == _var30)
    if _var31:
        _var32 = sess.run(W)
        _var33 = sess.run(b)
        _var34 = (step, _var32, _var33)
        print(_var34)
from keras.models import Sequential
from keras.layers import Dense, Activation
_var35 = 32
_var36 = 784
_var37 = (_var36,)
_var38 = Dense(_var35, input_shape=_var37)
_var39 = 'relu'
_var40 = Activation(_var39)
_var41 = 10
_var42 = Dense(_var41)
_var43 = 'softmax'
_var44 = Activation(_var43)
_var45 = [_var38, _var40, _var42, _var44]
model = Sequential(_var45)
model_0 = Sequential()
_var46 = 32
_var47 = 784
_var48 = Dense(_var46, input_dim=_var47)
model_0.add(_var48)
_var49 = 'relu'
_var50 = Activation(_var49)
model_0.add(_var50)
_var51 = 32
_var52 = 784
_var53 = (_var52,)
_var54 = 'relu'
_var55 = Dense(_var51, input_shape=_var53, activation=_var54)
_var56 = 10
_var57 = 'softmax'
_var58 = Dense(_var56, activation=_var57)
_var59 = [_var55, _var58]
model_1 = Sequential(_var59)
_var60 = 32
_var61 = 784
_var62 = (_var61,)
_var63 = Dense(_var60, input_shape=_var62)
_var64 = 'relu'
_var65 = Activation(_var64)
_var66 = 10
_var67 = Dense(_var66)
_var68 = 'softmax'
_var69 = Activation(_var68)
_var70 = [_var63, _var65, _var67, _var69]
model_2 = Sequential(_var70)
_var71 = 'adam'
_var72 = 'categorical_crossentropy'
_var73 = ['accuracy']
model_2.compile(_var71, _var72, metrics=_var73)
model_2.summary()
from keras.datasets import mnist
import keras
(_var74, _var75) = mnist.load_data()
_var78 = 0
X_train = _var74[_var78]
_var79 = 1
y_train = _var74[_var79]
_var82 = 0
X_test = _var75[_var82]
_var83 = 1
y_test = _var75[_var83]
_var84 = 60000
_var85 = 784
X_train_0 = X_train.reshape(_var84, _var85)
_var86 = 10000
_var87 = 784
X_test_0 = X_test.reshape(_var86, _var87)
_var88 = 'float32'
X_train_1 = X_train_0.astype(_var88)
_var89 = 'float32'
X_test_1 = X_test_0.astype(_var89)
_var90 = 255
X_train_2 = (X_train_1 / _var90)
_var91 = 255
X_test_2 = (X_test_1 / _var91)
_var92 = X_train_2.shape
_var93 = 0
_var94 = _var92[_var93]
_var95 = 'train samples'
_var96 = (_var94, _var95)
print(_var96)
_var97 = X_test_2.shape
_var98 = 0
_var99 = _var97[_var98]
_var100 = 'test samples'
_var101 = (_var99, _var100)
print(_var101)
num_classes = 10
_var102 = keras.utils
y_train_0 = _var102.to_categorical(y_train, num_classes)
_var103 = keras.utils
y_test_0 = _var103.to_categorical(y_test, num_classes)
_var104 = 128
_var105 = 10
_var106 = 1
model_3 = model_2.fit(X_train_2, y_train_0, batch_size=_var104, epochs=_var105, verbose=_var106)
_var107 = 0
score = model_3.evaluate(X_test_2, y_test_0, verbose=_var107)
_var108 = 'Test loss: {:.3f}'
_var109 = 0
_var110 = score[_var109]
_var111 = _var108.format(_var110)
print(_var111)
_var112 = 'Test Accuracy: {:.3f}'
_var113 = 1
_var114 = score[_var113]
_var115 = _var112.format(_var114)
print(_var115)
_var116 = 32
_var117 = 784
_var118 = (_var117,)
_var119 = Dense(_var116, input_shape=_var118)
_var120 = 'relu'
_var121 = Activation(_var120)
_var122 = 10
_var123 = Dense(_var122)
_var124 = 'softmax'
_var125 = Activation(_var124)
_var126 = [_var119, _var121, _var123, _var125]
model_4 = Sequential(_var126)
_var127 = 'adam'
_var128 = 'categorical_crossentropy'
_var129 = ['accuracy']
model_4.compile(_var127, _var128, metrics=_var129)
_var130 = 128
_var131 = 10
_var132 = 1
_var133 = 0.1
model_5 = model_4.fit(X_train_2, y_train_0, batch_size=_var130, epochs=_var131, verbose=_var132, validation_split=_var133)
_var134 = 32
_var135 = 784
_var136 = (_var135,)
_var137 = 'relu'
_var138 = Dense(_var134, input_shape=_var136, activation=_var137)
_var139 = 10
_var140 = 'softmax'
_var141 = Dense(_var139, activation=_var140)
_var142 = [_var138, _var141]
model_6 = Sequential(_var142)
_var143 = 'adam'
_var144 = 'categorical_crossentropy'
_var145 = ['accuracy']
model_6.compile(_var143, _var144, metrics=_var145)
_var146 = 128
_var147 = 100
_var148 = 1
_var149 = 0.1
history_callback = model_6.fit(X_train_2, y_train_0, batch_size=_var146, epochs=_var147, verbose=_var148, validation_split=_var149)
model_7 = history_callback

def plot_history(logger):
    _var150 = logger.history
    df = pd.DataFrame(_var150)
    _var151 = ['acc', 'val_acc']
    _var152 = df[_var151]
    _var152.plot()
    _var153 = 'accuracy'
    plt.ylabel(_var153)
    _var154 = ['loss', 'val_loss']
    _var155 = df[_var154]
    _var156 = '--'
    _var157 = plt.twinx()
    _var155.plot(linestyle=_var156, ax=_var157)
    _var158 = 'loss'
    plt.ylabel(_var158)
_var159 = history_callback.history
df_0 = pd.DataFrame(_var159)
_var160 = ['acc', 'val_acc']
_var161 = df_0[_var160]
_var161.plot()
_var162 = 'accuracy'
plt.ylabel(_var162)
_var163 = ['loss', 'val_loss']
_var164 = df_0[_var163]
_var165 = '--'
_var166 = plt.twinx()
_var164.plot(linestyle=_var165, ax=_var166)
_var167 = 'loss'
plt.ylabel(_var167)
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

def make_model(optimizer_0='adam', hidden_size=32):
    _var168 = 784
    _var169 = (_var168,)
    _var170 = Dense(hidden_size, input_shape=_var169)
    _var171 = 'relu'
    _var172 = Activation(_var171)
    _var173 = 10
    _var174 = Dense(_var173)
    _var175 = 'softmax'
    _var176 = Activation(_var175)
    _var177 = [_var170, _var172, _var174, _var176]
    model_8 = Sequential(_var177)
    _var178 = 'categorical_crossentropy'
    _var179 = ['accuracy']
    model_8.compile(optimizer=optimizer_0, loss=_var178, metrics=_var179)
    return model_8
clf = KerasClassifier(make_model)
_var180 = [1, 5, 10]
_var181 = [32, 64, 256]
param_grid = {'epochs': _var180, 'hidden_size': _var181}
_var182 = 5
grid = GridSearchCV(clf, param_grid=param_grid, cv=_var182)
grid_0 = grid.fit(X_train_2, y_train_0)
_var183 = grid_0.cv_results_
res = pd.DataFrame(_var183)
_var184 = ['param_epochs', 'param_hidden_size']
_var185 = ['mean_train_score', 'mean_test_score']
res.pivot_table(index=_var184, values=_var185)
_var186 = 1024
_var187 = 784
_var188 = (_var187,)
_var189 = 'relu'
_var190 = Dense(_var186, input_shape=_var188, activation=_var189)
_var191 = 1024
_var192 = 'relu'
_var193 = Dense(_var191, activation=_var192)
_var194 = 10
_var195 = 'softmax'
_var196 = Dense(_var194, activation=_var195)
_var197 = [_var190, _var193, _var196]
model_9 = Sequential(_var197)
_var198 = 'adam'
_var199 = 'categorical_crossentropy'
_var200 = ['accuracy']
model_9.compile(_var198, _var199, metrics=_var200)
_var201 = 128
_var202 = 20
_var203 = 1
_var204 = 0.1
history = model_9.fit(X_train_2, y_train_0, batch_size=_var201, epochs=_var202, verbose=_var203, validation_split=_var204)
model_10 = history
_var205 = 0
score_0 = model_10.evaluate(X_test_2, y_test_0, verbose=_var205)
score_0
model_10.summary()
_var206 = history.history
df_1 = pd.DataFrame(_var206)
_var207 = ['acc', 'val_acc']
_var208 = df_1[_var207]
_var208.plot()
_var209 = 'accuracy'
plt.ylabel(_var209)
_var210 = ['loss', 'val_loss']
_var211 = df_1[_var210]
_var212 = '--'
_var213 = plt.twinx()
_var211.plot(linestyle=_var212, ax=_var213)
_var214 = 'loss'
plt.ylabel(_var214)
from keras.layers import Dropout
_var215 = 1024
_var216 = 784
_var217 = (_var216,)
_var218 = 'relu'
_var219 = Dense(_var215, input_shape=_var217, activation=_var218)
_var220 = 0.5
_var221 = Dropout(_var220)
_var222 = 1024
_var223 = 'relu'
_var224 = Dense(_var222, activation=_var223)
_var225 = 0.5
_var226 = Dropout(_var225)
_var227 = 10
_var228 = 'softmax'
_var229 = Dense(_var227, activation=_var228)
_var230 = [_var219, _var221, _var224, _var226, _var229]
model_dropout = Sequential(_var230)
_var231 = 'adam'
_var232 = 'categorical_crossentropy'
_var233 = ['accuracy']
model_dropout.compile(_var231, _var232, metrics=_var233)
_var234 = 128
_var235 = 20
_var236 = 1
_var237 = 0.1
history_dropout = model_dropout.fit(X_train_2, y_train_0, batch_size=_var234, epochs=_var235, verbose=_var236, validation_split=_var237)
model_dropout_0 = history_dropout
_var238 = history_dropout.history
df_2 = pd.DataFrame(_var238)
_var239 = ['acc', 'val_acc']
_var240 = df_2[_var239]
_var240.plot()
_var241 = 'accuracy'
plt.ylabel(_var241)
_var242 = ['loss', 'val_loss']
_var243 = df_2[_var242]
_var244 = '--'
_var245 = plt.twinx()
_var243.plot(linestyle=_var244, ax=_var245)
_var246 = 'loss'
plt.ylabel(_var246)
_var247 = 0
score_1 = model_10.evaluate(X_test_2, y_test_0, verbose=_var247)
score_1
from keras.layers import BatchNormalization
_var248 = 512
_var249 = 784
_var250 = (_var249,)
_var251 = Dense(_var248, input_shape=_var250)
_var252 = BatchNormalization()
_var253 = 'relu'
_var254 = Activation(_var253)
_var255 = 512
_var256 = Dense(_var255)
_var257 = BatchNormalization()
_var258 = 'relu'
_var259 = Activation(_var258)
_var260 = 10
_var261 = 'softmax'
_var262 = Dense(_var260, activation=_var261)
_var263 = [_var251, _var252, _var254, _var256, _var257, _var259, _var262]
model_bn = Sequential(_var263)
_var264 = 'adam'
_var265 = 'categorical_crossentropy'
_var266 = ['accuracy']
model_bn.compile(_var264, _var265, metrics=_var266)
_var267 = 128
_var268 = 10
_var269 = 1
_var270 = 0.1
history_bn = model_10.fit(X_train_2, y_train_0, batch_size=_var267, epochs=_var268, verbose=_var269, validation_split=_var270)
model_11 = history_bn
plot_history(history)
from scipy.ndimage import convolve
_var271 = np.random
_var272 = 2
rng = _var271.RandomState(_var272)
_var273 = 200
_var274 = rng.normal(size=_var273)
signal = np.cumsum(_var274)
plt.plot(signal)
_var275 = (- 2)
_var276 = 2
_var277 = 15
_var278 = np.linspace(_var275, _var276, _var277)
_var279 = 2
_var280 = (_var278 ** _var279)
_var281 = (- _var280)
gaussian_filter = np.exp(_var281)
_var282 = gaussian_filter.sum()
gaussian_filter_0 = (gaussian_filter / _var282)
plt.plot(gaussian_filter_0)
gaussian_filter_0
plt.plot(signal)
_var283 = convolve(signal, gaussian_filter_0)
plt.plot(_var283)
from scipy.misc import imread
_var284 = 'IMG_20170207_090931.jpg'
image = imread(_var284)
plt.imshow(image)
_var285 = np.newaxis
_var286 = gaussian_filter_0[:, _var285]
gaussian_2d = (gaussian_filter_0 * _var286)
plt.matshow(gaussian_2d)
_var287 = np.newaxis
_var288 = gaussian_2d[:, :, _var287]
out = convolve(image, _var288)
plt.imshow(out)
_var289 = 2
gray_image = image.mean(axis=_var289)
_var290 = 'gray'
plt.imshow(gray_image, cmap=_var290)
_var291 = (- 1)
_var292 = [_var291, 1]
_var293 = [_var292]
gradient_2d = convolve(gaussian_2d, _var293)
plt.imshow(gradient_2d)
edges = convolve(gray_image, gradient_2d)
_var294 = 'gray'
plt.imshow(edges, cmap=_var294)
batch_size = 128
num_classes_0 = 10
epochs = 12
_var295 = 28
_var296 = 28
img_rows = _var295
img_cols = _var296
(_var297, _var298) = mnist.load_data()
_var301 = 0
x_train = _var297[_var301]
_var302 = 1
y_train_1 = _var297[_var302]
_var305 = 0
x_test = _var298[_var305]
_var306 = 1
y_test_1 = _var298[_var306]
_var307 = x_train.shape
_var308 = 0
_var309 = _var307[_var308]
_var310 = 1
X_train_images = x_train.reshape(_var309, img_rows, img_cols, _var310)
_var311 = x_test.shape
_var312 = 0
_var313 = _var311[_var312]
_var314 = 1
X_test_images = x_test.reshape(_var313, img_rows, img_cols, _var314)
_var315 = 1
input_shape = (img_rows, img_cols, _var315)
_var316 = keras.utils
y_train_2 = _var316.to_categorical(y_train_1, num_classes_0)
_var317 = keras.utils
y_test_2 = _var317.to_categorical(y_test_1, num_classes_0)
from keras.layers import Conv2D, MaxPooling2D, Flatten
num_classes_1 = 10
cnn = Sequential()
_var318 = 32
_var319 = 3
_var320 = 3
_var321 = (_var319, _var320)
_var322 = 'relu'
_var323 = Conv2D(_var318, kernel_size=_var321, activation=_var322, input_shape=input_shape)
cnn.add(_var323)
_var324 = 2
_var325 = 2
_var326 = (_var324, _var325)
_var327 = MaxPooling2D(pool_size=_var326)
cnn.add(_var327)
_var328 = 32
_var329 = 3
_var330 = 3
_var331 = (_var329, _var330)
_var332 = 'relu'
_var333 = Conv2D(_var328, _var331, activation=_var332)
cnn.add(_var333)
_var334 = 2
_var335 = 2
_var336 = (_var334, _var335)
_var337 = MaxPooling2D(pool_size=_var336)
cnn.add(_var337)
_var338 = Flatten()
cnn.add(_var338)
_var339 = 64
_var340 = 'relu'
_var341 = Dense(_var339, activation=_var340)
cnn.add(_var341)
_var342 = 'softmax'
_var343 = Dense(num_classes_1, activation=_var342)
cnn.add(_var343)
cnn.summary()
_var344 = 'adam'
_var345 = 'categorical_crossentropy'
_var346 = ['accuracy']
cnn.compile(_var344, _var345, metrics=_var346)
_var347 = 128
_var348 = 20
_var349 = 1
_var350 = 0.1
history_cnn = cnn.fit(X_train_images, y_train_2, batch_size=_var347, epochs=_var348, verbose=_var349, validation_split=_var350)
cnn_0 = history_cnn
plot_history(history_cnn)
cnn_0.evaluate(X_test_images, y_test_2)
_var351 = history_cnn.history
df_3 = pd.DataFrame(_var351)
_var352 = ['acc', 'val_acc']
_var353 = df_3[_var352]
_var353.plot()
_var354 = 'accuracy'
plt.ylabel(_var354)
_var355 = 0.9
_var356 = 1
plt.ylim(_var355, _var356)
_var357 = cnn_0.layers
_var358 = 0
layer1 = _var357[_var358]
(_var359, _var360) = layer1.get_weights()
weights = _var359
biases = _var360
weights.shape
_var363 = 4
_var364 = 6
(_var361, _var362) = plt.subplots(_var363, _var364)
fig = _var361
axes = _var362
_var365 = axes.ravel()
_var366 = weights.T
_var367 = zip(_var365, _var366)
for _var368 in _var367:
    _var371 = 0
    ax = _var368[_var371]
    _var372 = 1
    weight = _var368[_var372]
    _var373 = 0
    _var374 = weight[_var373, :, :]
    ax.imshow(_var374)
from keras.layers import Conv2D, MaxPooling2D, Flatten
num_classes_2 = 10
cnn_1 = Sequential()
_var375 = 8
_var376 = 5
_var377 = 5
_var378 = (_var376, _var377)
_var379 = 'relu'
_var380 = Conv2D(_var375, kernel_size=_var378, activation=_var379, input_shape=input_shape)
cnn_1.add(_var380)
_var381 = 2
_var382 = 2
_var383 = (_var381, _var382)
_var384 = MaxPooling2D(pool_size=_var383)
cnn_1.add(_var384)
_var385 = 8
_var386 = 5
_var387 = 5
_var388 = (_var386, _var387)
_var389 = 'relu'
_var390 = Conv2D(_var385, _var388, activation=_var389)
cnn_1.add(_var390)
_var391 = 2
_var392 = 2
_var393 = (_var391, _var392)
_var394 = MaxPooling2D(pool_size=_var393)
cnn_1.add(_var394)
_var395 = Flatten()
cnn_1.add(_var395)
_var396 = 64
_var397 = 'relu'
_var398 = Dense(_var396, activation=_var397)
cnn_1.add(_var398)
_var399 = 'softmax'
_var400 = Dense(num_classes_2, activation=_var399)
cnn_1.add(_var400)
_var401 = 'adam'
_var402 = 'categorical_crossentropy'
_var403 = ['accuracy']
cnn_1.compile(_var401, _var402, metrics=_var403)
_var404 = 128
_var405 = 10
_var406 = 1
_var407 = 0.1
history_cnn_0 = cnn_1.fit(X_train_images, y_train_2, batch_size=_var404, epochs=_var405, verbose=_var406, validation_split=_var407)
cnn_2 = history_cnn_0
_var410 = cnn_2.layers
_var411 = 0
_var412 = _var410[_var411]
(_var408, _var409) = _var412.get_weights()
weights_0 = _var408
biases_0 = _var409
_var415 = 2
_var416 = 4
(_var413, _var414) = plt.subplots(_var415, _var416)
fig_0 = _var413
axes_0 = _var414
_var417 = weights_0.min()
_var418 = weights_0.max()
mi = _var417
ma = _var418
_var419 = axes_0.ravel()
_var420 = weights_0.T
_var421 = zip(_var419, _var420)
for _var422 in _var421:
    _var425 = 0
    ax_0 = _var422[_var425]
    _var426 = 1
    weight_0 = _var422[_var426]
    _var427 = 0
    _var428 = weight_0[_var427, :, :]
    _var429 = _var428.T
    ax_0.imshow(_var429, vmin=mi, vmax=ma)
ax_1 = __phi__(ax_0, ax)
weight_1 = __phi__(weight_0, weight)
weights_0.shape
_var430 = 0
_var431 = 0
_var432 = weights_0[:, :, _var430, _var431]
plt.imshow(_var432)
_var433 = 0
asdf = cnn_2.get_input_at(_var433)
from keras import backend as K
_var434 = cnn_2.layers
_var435 = 0
_var436 = _var434[_var435]
_var437 = _var436.input
_var438 = [_var437]
_var439 = cnn_2.layers
_var440 = 0
_var441 = _var439[_var440]
_var442 = _var441.output
_var443 = [_var442]
get_1rd_layer_output = K.function(_var438, _var443)
_var444 = cnn_2.layers
_var445 = 0
_var446 = _var444[_var445]
_var447 = _var446.input
_var448 = [_var447]
_var449 = cnn_2.layers
_var450 = 3
_var451 = _var449[_var450]
_var452 = _var451.output
_var453 = [_var452]
get_3rd_layer_output = K.function(_var448, _var453)
_var454 = 5
_var455 = X_train_images[:_var454]
_var456 = [_var455]
_var457 = get_1rd_layer_output(_var456)
_var458 = 0
layer1_output = _var457[_var458]
_var459 = 5
_var460 = X_train_images[:_var459]
_var461 = [_var460]
_var462 = get_3rd_layer_output(_var461)
_var463 = 0
layer3_output = _var462[_var463]
layer1_output.shape
layer3_output.shape
_var466 = cnn_2.layers
_var467 = 0
_var468 = _var466[_var467]
(_var464, _var465) = _var468.get_weights()
weights_1 = _var464
biases_1 = _var465
_var469 = layer1_output.shape
_var470 = 0
n_images = _var469[_var470]
_var471 = layer1_output.shape
_var472 = 3
n_filters = _var471[_var472]
_var475 = 2
_var476 = (n_images * _var475)
_var477 = 1
_var478 = (n_filters + _var477)
_var479 = ()
_var480 = ()
_var481 = {'xticks': _var479, 'yticks': _var480}
(_var473, _var474) = plt.subplots(_var476, _var478, subplot_kw=_var481)
fig_1 = _var473
axes_1 = _var474
_var482 = layer1_output.shape
_var483 = 0
_var484 = _var482[_var483]
_var485 = range(_var484)
for i in _var485:
    _var486 = 2
    _var487 = (_var486 * i)
    _var488 = 0
    _var489 = (_var487, _var488)
    _var490 = axes_1[_var489]
    _var491 = 0
    _var492 = X_train_images[i, :, :, _var491]
    _var493 = 'gray_r'
    _var490.imshow(_var492, cmap=_var493)
    _var494 = 2
    _var495 = (_var494 * i)
    _var496 = 1
    _var497 = (_var495 + _var496)
    _var498 = 0
    _var499 = (_var497, _var498)
    _var500 = axes_1[_var499]
    _var501 = False
    _var500.set_visible(_var501)
    _var502 = layer1_output.shape
    _var503 = 3
    _var504 = _var502[_var503]
    _var505 = range(_var504)
    for j in _var505:
        _var506 = 2
        _var507 = (_var506 * i)
        _var508 = 1
        _var509 = (j + _var508)
        _var510 = (_var507, _var509)
        _var511 = axes_1[_var510]
        _var512 = layer1_output[i, :, :, j]
        _var513 = 'gray_r'
        _var511.imshow(_var512, cmap=_var513)
        _var514 = 2
        _var515 = (_var514 * i)
        _var516 = 1
        _var517 = (_var515 + _var516)
        _var518 = 1
        _var519 = (j + _var518)
        _var520 = (_var517, _var519)
        _var521 = axes_1[_var520]
        _var522 = layer3_output[i, :, :, j]
        _var523 = 'gray_r'
        _var521.imshow(_var522, cmap=_var523)
from keras.layers import Conv2D, MaxPooling2D, Flatten
num_classes_3 = 10
cnn_small = Sequential()
_var524 = 8
_var525 = 3
_var526 = 3
_var527 = (_var525, _var526)
_var528 = 'relu'
_var529 = Conv2D(_var524, kernel_size=_var527, activation=_var528, input_shape=input_shape)
cnn_small.add(_var529)
_var530 = 2
_var531 = 2
_var532 = (_var530, _var531)
_var533 = MaxPooling2D(pool_size=_var532)
cnn_small.add(_var533)
_var534 = 8
_var535 = 3
_var536 = 3
_var537 = (_var535, _var536)
_var538 = 'relu'
_var539 = Conv2D(_var534, _var537, activation=_var538)
cnn_small.add(_var539)
_var540 = 2
_var541 = 2
_var542 = (_var540, _var541)
_var543 = MaxPooling2D(pool_size=_var542)
cnn_small.add(_var543)
_var544 = Flatten()
cnn_small.add(_var544)
_var545 = 64
_var546 = 'relu'
_var547 = Dense(_var545, activation=_var546)
cnn_small.add(_var547)
_var548 = 'softmax'
_var549 = Dense(num_classes_3, activation=_var548)
cnn_small.add(_var549)
cnn_small.summary()
_var550 = 'adam'
_var551 = 'categorical_crossentropy'
_var552 = ['accuracy']
cnn_small.compile(_var550, _var551, metrics=_var552)
_var553 = 128
_var554 = 10
_var555 = 1
_var556 = 0.1
history_cnn_small = cnn_small.fit(X_train_images, y_train_2, batch_size=_var553, epochs=_var554, verbose=_var555, validation_split=_var556)
cnn_small_0 = history_cnn_small
_var559 = cnn_small_0.layers
_var560 = 0
_var561 = _var559[_var560]
(_var557, _var558) = _var561.get_weights()
weights_2 = _var557
biases_2 = _var558
_var564 = cnn_small_0.layers
_var565 = 2
_var566 = _var564[_var565]
(_var562, _var563) = _var566.get_weights()
weights2 = _var562
biases2 = _var563
_var567 = weights_2.shape
print(_var567)
_var568 = weights2.shape
print(_var568)
_var571 = 9
_var572 = 8
_var573 = 10
_var574 = 8
_var575 = (_var573, _var574)
_var576 = ()
_var577 = ()
_var578 = {'xticks': _var576, 'yticks': _var577}
(_var569, _var570) = plt.subplots(_var571, _var572, figsize=_var575, subplot_kw=_var578)
fig_2 = _var569
axes_2 = _var570
_var579 = weights_2.min()
_var580 = weights_2.max()
mi_0 = _var579
ma_0 = _var580
_var581 = 0
_var582 = axes_2[_var581]
_var583 = weights_2.T
_var584 = zip(_var582, _var583)
for _var585 in _var584:
    _var588 = 0
    ax_2 = _var585[_var588]
    _var589 = 1
    weight_2 = _var585[_var589]
    _var590 = 0
    _var591 = weight_2[_var590, :, :]
    _var592 = _var591.T
    ax_2.imshow(_var592, vmin=mi_0, vmax=ma_0)
ax_3 = __phi__(ax_2, ax_1)
weight_3 = __phi__(weight_2, weight_1)
_var593 = 0
_var594 = 0
_var595 = (_var593, _var594)
_var596 = axes_2[_var595]
_var597 = 'layer1'
_var596.set_ylabel(_var597)
_var598 = weights2.min()
_var599 = weights2.max()
mi_1 = _var598
ma_1 = _var599
_var600 = 1
_var601 = 9
_var602 = range(_var600, _var601)
for i_0 in _var602:
    _var603 = 0
    _var604 = (i_0, _var603)
    _var605 = axes_2[_var604]
    _var606 = 'layer3'
    _var605.set_ylabel(_var606)
_var607 = 1
_var608 = axes_2[_var607:]
_var609 = _var608.ravel()
_var610 = 3
_var611 = 3
_var612 = (- 1)
_var613 = weights2.reshape(_var610, _var611, _var612)
_var614 = _var613.T
_var615 = zip(_var609, _var614)
for _var616 in _var615:
    _var619 = 0
    ax_4 = _var616[_var619]
    _var620 = 1
    weight_4 = _var616[_var620]
    _var621 = weight_4[:, :]
    _var622 = _var621.T
    ax_4.imshow(_var622, vmin=mi_1, vmax=ma_1)
ax_5 = __phi__(ax_4, ax_3)
weight_5 = __phi__(weight_4, weight_3)
from keras import backend as K
_var623 = cnn_small_0.layers
_var624 = 0
_var625 = _var623[_var624]
_var626 = _var625.input
_var627 = [_var626]
_var628 = cnn_small_0.layers
_var629 = 0
_var630 = _var628[_var629]
_var631 = _var630.output
_var632 = [_var631]
get_1rd_layer_output_0 = K.function(_var627, _var632)
_var633 = cnn_small_0.layers
_var634 = 0
_var635 = _var633[_var634]
_var636 = _var635.input
_var637 = [_var636]
_var638 = cnn_small_0.layers
_var639 = 2
_var640 = _var638[_var639]
_var641 = _var640.output
_var642 = [_var641]
get_3rd_layer_output_0 = K.function(_var637, _var642)
_var643 = 5
_var644 = X_train_images[:_var643]
_var645 = [_var644]
_var646 = get_1rd_layer_output_0(_var645)
_var647 = 0
layer1_output_0 = _var646[_var647]
_var648 = 5
_var649 = X_train_images[:_var648]
_var650 = [_var649]
_var651 = get_3rd_layer_output_0(_var650)
_var652 = 0
layer3_output_0 = _var651[_var652]
layer1_output_0.shape
layer3_output_0.shape
_var655 = cnn_2.layers
_var656 = 0
_var657 = _var655[_var656]
(_var653, _var654) = _var657.get_weights()
weights_3 = _var653
biases_3 = _var654
_var658 = layer1_output_0.shape
_var659 = 0
n_images_0 = _var658[_var659]
_var660 = layer1_output_0.shape
_var661 = 3
n_filters_0 = _var660[_var661]
_var664 = 2
_var665 = (n_images_0 * _var664)
_var666 = 1
_var667 = (n_filters_0 + _var666)
_var668 = 10
_var669 = 8
_var670 = (_var668, _var669)
_var671 = ()
_var672 = ()
_var673 = {'xticks': _var671, 'yticks': _var672}
(_var662, _var663) = plt.subplots(_var665, _var667, figsize=_var670, subplot_kw=_var673)
fig_3 = _var662
axes_3 = _var663
_var674 = layer1_output_0.shape
_var675 = 0
_var676 = _var674[_var675]
_var677 = range(_var676)
for i_1 in _var677:
    _var678 = 2
    _var679 = (_var678 * i_1)
    _var680 = 0
    _var681 = (_var679, _var680)
    _var682 = axes_3[_var681]
    _var683 = 0
    _var684 = X_train_images[i_1, :, :, _var683]
    _var685 = 'gray_r'
    _var682.imshow(_var684, cmap=_var685)
    _var686 = 2
    _var687 = (_var686 * i_1)
    _var688 = 1
    _var689 = (_var687 + _var688)
    _var690 = 0
    _var691 = (_var689, _var690)
    _var692 = axes_3[_var691]
    _var693 = False
    _var692.set_visible(_var693)
    _var694 = 2
    _var695 = (_var694 * i_1)
    _var696 = 1
    _var697 = (_var695, _var696)
    _var698 = axes_3[_var697]
    _var699 = 'layer1'
    _var698.set_ylabel(_var699)
    _var700 = 2
    _var701 = (_var700 * i_1)
    _var702 = 1
    _var703 = (_var701 + _var702)
    _var704 = 1
    _var705 = (_var703, _var704)
    _var706 = axes_3[_var705]
    _var707 = 'layer3'
    _var706.set_ylabel(_var707)
    _var708 = layer1_output_0.shape
    _var709 = 3
    _var710 = _var708[_var709]
    _var711 = range(_var710)
    for j_0 in _var711:
        _var712 = 2
        _var713 = (_var712 * i_1)
        _var714 = 1
        _var715 = (j_0 + _var714)
        _var716 = (_var713, _var715)
        _var717 = axes_3[_var716]
        _var718 = layer1_output_0[i_1, :, :, j_0]
        _var719 = 'gray_r'
        _var717.imshow(_var718, cmap=_var719)
        _var720 = 2
        _var721 = (_var720 * i_1)
        _var722 = 1
        _var723 = (_var721 + _var722)
        _var724 = 1
        _var725 = (j_0 + _var724)
        _var726 = (_var723, _var725)
        _var727 = axes_3[_var726]
        _var728 = layer3_output_0[i_1, :, :, j_0]
        _var729 = 'gray_r'
        _var727.imshow(_var728, cmap=_var729)
j_1 = __phi__(j_0, j)
from keras.layers import BatchNormalization
num_classes_4 = 10
cnn_small_bn = Sequential()
_var730 = 8
_var731 = 3
_var732 = 3
_var733 = (_var731, _var732)
_var734 = Conv2D(_var730, kernel_size=_var733, input_shape=input_shape)
cnn_small_bn.add(_var734)
_var735 = 'relu'
_var736 = Activation(_var735)
cnn_small_bn.add(_var736)
_var737 = BatchNormalization()
cnn_small_bn.add(_var737)
_var738 = 2
_var739 = 2
_var740 = (_var738, _var739)
_var741 = MaxPooling2D(pool_size=_var740)
cnn_small_bn.add(_var741)
_var742 = 8
_var743 = 3
_var744 = 3
_var745 = (_var743, _var744)
_var746 = Conv2D(_var742, _var745)
cnn_small_bn.add(_var746)
_var747 = 'relu'
_var748 = Activation(_var747)
cnn_small_bn.add(_var748)
_var749 = BatchNormalization()
cnn_small_bn.add(_var749)
_var750 = 2
_var751 = 2
_var752 = (_var750, _var751)
_var753 = MaxPooling2D(pool_size=_var752)
cnn_small_bn.add(_var753)
_var754 = Flatten()
cnn_small_bn.add(_var754)
_var755 = 64
_var756 = 'relu'
_var757 = Dense(_var755, activation=_var756)
cnn_small_bn.add(_var757)
_var758 = 'softmax'
_var759 = Dense(num_classes_4, activation=_var758)
cnn_small_bn.add(_var759)
_var760 = 'adam'
_var761 = 'categorical_crossentropy'
_var762 = ['accuracy']
cnn_small_bn.compile(_var760, _var761, metrics=_var762)
_var763 = 128
_var764 = 10
_var765 = 1
_var766 = 0.1
history_cnn_small_bn = cnn_small_bn.fit(X_train_images, y_train_2, batch_size=_var763, epochs=_var764, verbose=_var765, validation_split=_var766)
cnn_small_bn_0 = history_cnn_small_bn
_var767 = history_cnn_small_bn.history
hist_small_bn = pd.DataFrame(_var767)
_var768 = history_cnn_small.history
hist_small = pd.DataFrame(_var768)

def _func0(x):
    _var769 = ' BN'
    _var770 = (x + _var769)
    return _var770
_var771 = True
hist_small_bn.rename(columns=_func0, inplace=_var771)
_var772 = ['acc BN', 'val_acc BN']
_var773 = hist_small_bn[_var772]
_var773.plot()
_var774 = ['acc', 'val_acc']
_var775 = hist_small[_var774]
_var776 = plt.gca()
_var777 = '--'
_var778 = plt.cm
_var779 = 0
_var780 = _var778.Vega10(_var779)
_var781 = plt.cm
_var782 = 1
_var783 = _var781.Vega10(_var782)
_var784 = [_var780, _var783]
_var775.plot(ax=_var776, linestyle=_var777, color=_var784)
from keras.layers import BatchNormalization
num_classes_5 = 10
cnn32 = Sequential()
_var785 = 32
_var786 = 3
_var787 = 3
_var788 = (_var786, _var787)
_var789 = Conv2D(_var785, kernel_size=_var788, input_shape=input_shape)
cnn32.add(_var789)
_var790 = 'relu'
_var791 = Activation(_var790)
cnn32.add(_var791)
_var792 = 2
_var793 = 2
_var794 = (_var792, _var793)
_var795 = MaxPooling2D(pool_size=_var794)
cnn32.add(_var795)
_var796 = 32
_var797 = 3
_var798 = 3
_var799 = (_var797, _var798)
_var800 = Conv2D(_var796, _var799)
cnn32.add(_var800)
_var801 = 'relu'
_var802 = Activation(_var801)
cnn32.add(_var802)
_var803 = 2
_var804 = 2
_var805 = (_var803, _var804)
_var806 = MaxPooling2D(pool_size=_var805)
cnn32.add(_var806)
_var807 = Flatten()
cnn32.add(_var807)
_var808 = 64
_var809 = 'relu'
_var810 = Dense(_var808, activation=_var809)
cnn32.add(_var810)
_var811 = 'softmax'
_var812 = Dense(num_classes_5, activation=_var811)
cnn32.add(_var812)
_var813 = 'adam'
_var814 = 'categorical_crossentropy'
_var815 = ['accuracy']
cnn32.compile(_var813, _var814, metrics=_var815)
_var816 = 128
_var817 = 10
_var818 = 1
_var819 = 0.1
history_cnn_32 = cnn32.fit(X_train_images, y_train_2, batch_size=_var816, epochs=_var817, verbose=_var818, validation_split=_var819)
cnn32_0 = history_cnn_32
from keras.layers import BatchNormalization
num_classes_6 = 10
cnn32_bn = Sequential()
_var820 = 32
_var821 = 3
_var822 = 3
_var823 = (_var821, _var822)
_var824 = Conv2D(_var820, kernel_size=_var823, input_shape=input_shape)
cnn32_bn.add(_var824)
_var825 = 'relu'
_var826 = Activation(_var825)
cnn32_bn.add(_var826)
_var827 = BatchNormalization()
cnn32_bn.add(_var827)
_var828 = 2
_var829 = 2
_var830 = (_var828, _var829)
_var831 = MaxPooling2D(pool_size=_var830)
cnn32_bn.add(_var831)
_var832 = 32
_var833 = 3
_var834 = 3
_var835 = (_var833, _var834)
_var836 = Conv2D(_var832, _var835)
cnn32_bn.add(_var836)
_var837 = 'relu'
_var838 = Activation(_var837)
cnn32_bn.add(_var838)
_var839 = BatchNormalization()
cnn32_bn.add(_var839)
_var840 = 2
_var841 = 2
_var842 = (_var840, _var841)
_var843 = MaxPooling2D(pool_size=_var842)
cnn32_bn.add(_var843)
_var844 = Flatten()
cnn32_bn.add(_var844)
_var845 = 64
_var846 = 'relu'
_var847 = Dense(_var845, activation=_var846)
cnn32_bn.add(_var847)
_var848 = 'softmax'
_var849 = Dense(num_classes_6, activation=_var848)
cnn32_bn.add(_var849)
_var850 = 'adam'
_var851 = 'categorical_crossentropy'
_var852 = ['accuracy']
cnn32_bn.compile(_var850, _var851, metrics=_var852)
_var853 = 128
_var854 = 10
_var855 = 1
_var856 = 0.1
history_cnn_32_bn = cnn32_bn.fit(X_train_images, y_train_2, batch_size=_var853, epochs=_var854, verbose=_var855, validation_split=_var856)
cnn32_bn_0 = history_cnn_32_bn
_var857 = history_cnn_32_bn.history
hist_32_bn = pd.DataFrame(_var857)
_var858 = history_cnn_32.history
hist_32 = pd.DataFrame(_var858)

def _func1(x_0):
    _var859 = ' BN'
    _var860 = (x_0 + _var859)
    return _var860
_var861 = True
hist_32_bn.rename(columns=_func1, inplace=_var861)
_var862 = ['acc BN', 'val_acc BN']
_var863 = hist_32_bn[_var862]
_var863.plot()
_var864 = ['acc', 'val_acc']
_var865 = hist_32[_var864]
_var866 = plt.gca()
_var867 = '--'
_var868 = plt.cm
_var869 = 0
_var870 = _var868.Vega10(_var869)
_var871 = plt.cm
_var872 = 1
_var873 = _var871.Vega10(_var872)
_var874 = [_var870, _var873]
_var865.plot(ax=_var866, linestyle=_var867, color=_var874)
_var875 = 0.8
_var876 = 1
plt.ylim(_var875, _var876)
from keras import applications
_var877 = False
_var878 = 'imagenet'
model_12 = applications.VGG16(include_top=_var877, weights=_var878)
model_12.summary()
_var881 = model_12.layers
_var882 = 1
_var883 = _var881[_var882]
(_var879, _var880) = _var883.get_weights()
vgg_weights = _var879
vgg_biases = _var880
vgg_weights.shape
_var886 = 8
_var887 = 8
_var888 = 10
_var889 = 8
_var890 = (_var888, _var889)
_var891 = ()
_var892 = ()
_var893 = {'xticks': _var891, 'yticks': _var892}
(_var884, _var885) = plt.subplots(_var886, _var887, figsize=_var890, subplot_kw=_var893)
fig_4 = _var884
axes_4 = _var885
_var894 = vgg_weights.min()
_var895 = vgg_weights.max()
mi_2 = _var894
ma_2 = _var895
_var896 = axes_4.ravel()
_var897 = vgg_weights.T
_var898 = zip(_var896, _var897)
for _var899 in _var898:
    _var902 = 0
    ax_6 = _var899[_var902]
    _var903 = 1
    weight_6 = _var899[_var903]
    _var904 = weight_6.T
    ax_6.imshow(_var904)
ax_7 = __phi__(ax_6, ax_5)
weight_7 = __phi__(weight_6, weight_5)
plt.imshow(image)
_var905 = model_12.layers
_var906 = 0
_var907 = _var905[_var906]
_var908 = _var907.input
_var909 = [_var908]
_var910 = model_12.layers
_var911 = 3
_var912 = _var910[_var911]
_var913 = _var912.output
_var914 = [_var913]
get_3rd_layer_output_1 = K.function(_var909, _var914)
_var915 = model_12.layers
_var916 = 0
_var917 = _var915[_var916]
_var918 = _var917.input
_var919 = [_var918]
_var920 = model_12.layers
_var921 = 6
_var922 = _var920[_var921]
_var923 = _var922.output
_var924 = [_var923]
get_6rd_layer_output = K.function(_var919, _var924)
_var925 = [image]
_var926 = [_var925]
_var927 = get_3rd_layer_output_1(_var926)
_var928 = 0
layer3_output_1 = _var927[_var928]
_var929 = [image]
_var930 = [_var929]
_var931 = get_6rd_layer_output(_var930)
_var932 = 0
layer6_output = _var931[_var932]
_var933 = layer3_output_1.shape
print(_var933)
_var934 = layer6_output.shape
print(_var934)
_var937 = 2
_var938 = 8
_var939 = 10
_var940 = 4
_var941 = (_var939, _var940)
_var942 = ()
_var943 = ()
_var944 = {'xticks': _var942, 'yticks': _var943}
(_var935, _var936) = plt.subplots(_var937, _var938, figsize=_var941, subplot_kw=_var944)
fig_5 = _var935
axes_5 = _var936
_var945 = axes_5.ravel()
_var946 = layer3_output_1.T
_var947 = zip(_var945, _var946)
for _var948 in _var947:
    _var951 = 0
    ax_8 = _var948[_var951]
    _var952 = 1
    activation = _var948[_var952]
    _var953 = 0
    _var954 = activation[:, :, _var953]
    _var955 = _var954.T
    _var956 = 'gray_r'
    ax_8.imshow(_var955, cmap=_var956)
ax_9 = __phi__(ax_8, ax_7)
_var957 = 'after first pooling layer'
plt.suptitle(_var957)
_var960 = 2
_var961 = 8
_var962 = 10
_var963 = 4
_var964 = (_var962, _var963)
_var965 = ()
_var966 = ()
_var967 = {'xticks': _var965, 'yticks': _var966}
(_var958, _var959) = plt.subplots(_var960, _var961, figsize=_var964, subplot_kw=_var967)
fig_6 = _var958
axes_6 = _var959
_var968 = axes_6.ravel()
_var969 = layer6_output.T
_var970 = zip(_var968, _var969)
for _var971 in _var970:
    _var974 = 0
    ax_10 = _var971[_var974]
    _var975 = 1
    activation_0 = _var971[_var975]
    _var976 = 0
    _var977 = activation_0[:, :, _var976]
    _var978 = _var977.T
    _var979 = 'gray_r'
    ax_10.imshow(_var978, cmap=_var979)
activation_1 = __phi__(activation_0, activation)
ax_11 = __phi__(ax_10, ax_9)
_var980 = 'after second pooling layer'
plt.suptitle(_var980)
import flickrapi
import json
api_key = 'f770a9e7064fa7f8754b1ed8cc8cda4f'
api_secret = ' 2e750f2d723350c8 '
import flickrapi
_var981 = 'json'
flickr = flickrapi.FlickrAPI(api_key, api_secret, format=_var981)
_var982 = flickr.photos
_var983 = _var982.licenses
_var984 = _var983.getInfo()
_var985 = 'utf-8'
_var986 = _var984.decode(_var985)
json.loads(_var986)

def get_url(photo_id='33510015330'):
    _var987 = global_wrapper(flickr)
    _var988 = _var987.photos
    response = _var988.getsizes(photo_id=photo_id)
    _var989 = 'utf-8'
    _var990 = response.decode(_var989)
    _var991 = json.loads(_var990)
    _var992 = 'sizes'
    _var993 = _var991[_var992]
    _var994 = 'size'
    sizes = _var993[_var994]
    for size in sizes:
        _var995 = 'label'
        _var996 = size[_var995]
        _var997 = 'Small'
        _var998 = (_var996 == _var997)
        if _var998:
            _var999 = 'source'
            _var1000 = size[_var999]
            return _var1000
get_url()
from IPython.display import HTML
_var1001 = "<img src='https://farm4.staticflickr.com/3803/33510015330_d1fc801d16_m.jpg'>"
HTML(_var1001)

def search_ids(search_string='python', per_page=10):
    _var1002 = global_wrapper(flickr)
    _var1003 = _var1002.photos
    _var1004 = 'relevance'
    photos_response = _var1003.search(text=search_string, per_page=per_page, sort=_var1004)
    _var1005 = 'utf-8'
    _var1006 = photos_response.decode(_var1005)
    _var1007 = json.loads(_var1006)
    _var1008 = 'photos'
    _var1009 = _var1007[_var1008]
    _var1010 = 'photo'
    photos = _var1009[_var1010]
    ids = [photo['id'] for photo in photos]
    return ids
_var1011 = 'ball snake'
_var1012 = 100
ids_0 = search_ids(_var1011, per_page=_var1012)
urls_ball = [get_url(photo_id=i) for i in ids]
_var1013 = '\n'
_var1014 = ["<img src='{}'>".format(url) for url in urls_ball]
img_string = _var1013.join(_var1014)
HTML(img_string)
_var1015 = 'carpet python'
_var1016 = 100
ids_1 = search_ids(_var1015, per_page=_var1016)
urls_carpet = [get_url(photo_id=i) for i in ids]
_var1017 = '\n'
_var1018 = ["<img src='{}'>".format(url) for url in urls_carpet]
img_string_0 = _var1017.join(_var1018)
HTML(img_string_0)
_var1019 = get_ipython()
_var1020 = 'mkdir -p snakes/carpet'
_var1019.system(_var1020)
_var1021 = get_ipython()
_var1022 = 'mkdir snakes/ball'
_var1021.system(_var1022)
from urllib.request import urlretrieve
import os
for url in urls_carpet:
    _var1023 = os.path
    _var1024 = 'snakes'
    _var1025 = 'carpet'
    _var1026 = os.path
    _var1027 = _var1026.basename(url)
    _var1028 = _var1023.join(_var1024, _var1025, _var1027)
    urlretrieve(url, _var1028)
for url_0 in urls_ball:
    _var1029 = os.path
    _var1030 = 'snakes'
    _var1031 = 'ball'
    _var1032 = os.path
    _var1033 = _var1032.basename(url_0)
    _var1034 = _var1029.join(_var1030, _var1031, _var1033)
    urlretrieve(url_0, _var1034)
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
images_carpet = [image.load_img(os.path.join('snakes', 'carpet', os.path.basename(url)), target_size=(224, 224)) for url in urls_carpet]
images_ball = [image.load_img(os.path.join('snakes', 'ball', os.path.basename(url)), target_size=(224, 224)) for url in urls_ball]
_var1035 = [image.img_to_array(img) for img in (images_carpet + images_ball)]
X = np.array(_var1035)
from keras.preprocessing import image
images_carpet_0 = [image.load_img(os.path.join('snakes', 'carpet', os.path.basename(url)), target_size=(224, 224)) for url in urls_carpet]
images_ball_0 = [image.load_img(os.path.join('snakes', 'ball', os.path.basename(url)), target_size=(224, 224)) for url in urls_ball]
_var1036 = [image.img_to_array(img) for img in (images_carpet + images_ball)]
X_0 = np.array(_var1036)
_var1039 = 6
_var1040 = 4
_var1041 = ()
_var1042 = ()
_var1043 = {'xticks': _var1041, 'yticks': _var1042}
_var1044 = 5
_var1045 = 8
_var1046 = (_var1044, _var1045)
(_var1037, _var1038) = plt.subplots(_var1039, _var1040, subplot_kw=_var1043, figsize=_var1046)
fig_7 = _var1037
axes_7 = _var1038
_var1047 = axes_7.ravel()
_var1048 = zip(images_carpet_0, _var1047)
for _var1049 in _var1048:
    _var1052 = 0
    img = _var1049[_var1052]
    _var1053 = 1
    ax_12 = _var1049[_var1053]
    ax_12.imshow(img)
ax_13 = __phi__(ax_12, ax_11)
_var1056 = 6
_var1057 = 4
_var1058 = ()
_var1059 = ()
_var1060 = {'xticks': _var1058, 'yticks': _var1059}
_var1061 = 5
_var1062 = 8
_var1063 = (_var1061, _var1062)
(_var1054, _var1055) = plt.subplots(_var1056, _var1057, subplot_kw=_var1060, figsize=_var1063)
fig_8 = _var1054
axes_8 = _var1055
_var1064 = axes_8.ravel()
_var1065 = zip(images_ball_0, _var1064)
for _var1066 in _var1065:
    _var1069 = 0
    img_0 = _var1066[_var1069]
    _var1070 = 1
    ax_14 = _var1066[_var1070]
    ax_14.imshow(img_0)
img_1 = __phi__(img_0, img)
ax_15 = __phi__(ax_14, ax_13)
X_0.shape
from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X_0)
features = model_12.predict(X_pre)
features.shape
_var1071 = 200
_var1072 = (- 1)
features_ = features.reshape(_var1071, _var1072)
from sklearn.model_selection import train_test_split
_var1073 = 200
_var1074 = 'int'
y_0 = np.zeros(_var1073, dtype=_var1074)
_var1075 = 100
_var1076 = 1
y_1 = set_index_wrapper(y_0, slice(_var1075, None, None), _var1076)
(_var1077, _var1078, _var1079, _var1080) = train_test_split(features_, y_1, stratify=y_1)
X_train_3 = _var1077
X_test_3 = _var1078
y_train_3 = _var1079
y_test_3 = _var1080
from sklearn.linear_model import LogisticRegressionCV
_var1081 = LogisticRegressionCV()
lr = _var1081.fit(X_train_3, y_train_3)
_var1081_0 = lr
_var1082 = lr.score(X_train_3, y_train_3)
print(_var1082)
_var1083 = lr.score(X_test_3, y_test_3)
print(_var1083)
from sklearn.metrics import confusion_matrix
_var1084 = lr.predict(X_test_3)
confusion_matrix(y_test_3, _var1084)
