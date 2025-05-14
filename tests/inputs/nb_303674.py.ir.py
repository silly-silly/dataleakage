

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
_var0 = 0.8
_var1 = 0.01
_var2 = (_var0 * _var1)
_var3 = 0.8
_var4 = 0.01
_var5 = (_var3 * _var4)
_var6 = 0.096
_var7 = 0.99
_var8 = (_var6 * _var7)
_var9 = (_var5 + _var8)
(_var2 / _var9)
_var10 = get_ipython()
_var11 = 'matplotlib'
_var12 = 'inline'
_var10.run_line_magic(_var11, _var12)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import naive_bayes
_var13 = 'display.width'
_var14 = 500
pd.set_option(_var13, _var14)
_var15 = 'display.max_columns'
_var16 = 30
pd.set_option(_var15, _var16)
_var17 = '../../DAT18NYC/data/rt_critics.csv'
critics = pd.read_csv(_var17)
from sklearn.feature_extraction.text import CountVectorizer
text = ['Math is great', 'Math is really great', 'Exciting exciting Math']
_var18 = get_ipython()
_var19 = 'pinfo'
_var20 = 'CountVectorizer'
_var18.run_line_magic(_var19, _var20)
_var21 = 1
_var22 = 2
_var23 = (_var21, _var22)
vectorizer = CountVectorizer(ngram_range=_var23)
vectorizer_0 = vectorizer.fit(text)
_var24 = vectorizer_0.get_feature_names()
print(_var24)
x = vectorizer_0.transform(text)
_var25 = 'Sparse Matrix'
print(_var25)
print(x)
_var26 = type(x)
print(_var26)
print()
_var27 = 'Matrix'
print(_var27)
x_back = x.toarray()
print(x_back)
_var28 = vectorizer_0.get_feature_names()
pd.DataFrame(x_back, columns=_var28)
_var29 = critics.quote
_var30 = 2
_var31 = _var29[_var30]
print(_var31)
_var32 = critics.quote
rotten_vectorizer = vectorizer_0.fit(_var32)
vectorizer_1 = rotten_vectorizer
_var33 = critics.quote
x_0 = vectorizer_1.fit_transform(_var33)
critics.head()
_var34 = critics.fresh
_var35 = 'fresh'
_var36 = (_var34 == _var35)
_var37 = _var36.values
y = _var37.astype(int)

def train_and_measure(classifier, x_1, y_0, test_size):
    from sklearn import cross_validation
    _var42 = 0.2
    _var43 = 1234
    (_var38, _var39, _var40, _var41) = cross_validation.train_test_split(x_1, y_0, test_size=_var42, random_state=_var43)
    xtrain = _var38
    xtest = _var39
    ytrain = _var40
    ytest = _var41
    clf = classifier.fit(xtrain, ytrain)
    classifier_0 = clf
    training_accuracy = clf.score(xtrain, ytrain)
    test_accuracy = clf.score(xtest, ytest)
    print(classifier_0)
    _var44 = 'Accuracy on training data: %0.2f'
    _var45 = (_var44 % training_accuracy)
    print(_var45)
    _var46 = 'Accuracy on test data: %0.2f'
    _var47 = (_var46 % test_accuracy)
    print(_var47)
_var48 = naive_bayes.MultinomialNB()
_var49 = 0.2
train_and_measure(_var48, x_0, y, _var49)
_var50 = 1
x_ones = (x_0 > _var50)
_var51 = naive_bayes.BernoulliNB()
_var52 = 0.2
train_and_measure(_var51, x_ones, y, _var52)
from sklearn import linear_model
_var53 = linear_model.LogisticRegression()
_var54 = 0.2
train_and_measure(_var53, x_0, y, _var54)

def kfold_average_sd(classifier_1, n, x_2, y_1, return_plot=False):
    import numpy as np
    from sklearn import cross_validation
    _var55 = x_2.shape
    _var56 = 0
    _var57 = _var55[_var56]
    _var58 = 1234
    kfold = cross_validation.KFold(n=_var57, n_folds=n, random_state=_var58)
    train_acc = []
    test_acc = []
    for _var59 in kfold:
        _var62 = 0
        train_index = _var59[_var62]
        _var63 = 1
        test_index = _var59[_var63]
        _var64 = x_2[train_index]
        _var65 = y_1[train_index]
        clf_0 = classifier_1.fit(_var64, _var65)
        classifier_2 = clf_0
        _var66 = x_2[train_index]
        _var67 = y_1[train_index]
        _var68 = clf_0.score(_var66, _var67)
        train_acc.append(_var68)
        _var69 = x_2[test_index]
        _var70 = y_1[test_index]
        _var71 = clf_0.score(_var69, _var70)
        test_acc.append(_var71)
    classifier_3 = __phi__(classifier_2, classifier_1)
    if return_plot:
        plt.figure()
        _var72 = np.random
        _var73 = np.array(test_acc)
        _var74 = _var73.mean()
        _var75 = np.array(test_acc)
        _var76 = _var75.std()
        _var77 = 10000
        _var78 = _var72.normal(loc=_var74, scale=_var76, size=_var77)
        _var79 = True
        sns.kdeplot(_var78, shade=_var79)
    _var80 = np.array(test_acc)
    _var81 = _var80.mean()
    _var82 = np.array(test_acc)
    _var83 = _var82.std()
    return (_var81, _var83)
_var84 = naive_bayes.MultinomialNB()
_var85 = 5
_var86 = True
kfold_average_sd(_var84, _var85, x_0, y, _var86)

def find_k(classifier_4, x_3, y_2, max_num_k):
    from sklearn import cross_validation
    import numpy as np
    k_train_acc = []
    k_test_acc = []
    _var87 = 2
    _var88 = range(_var87, max_num_k)
    for i in _var88:
        _var89 = x_3.shape
        _var90 = 0
        _var91 = _var89[_var90]
        _var92 = True
        _var93 = 1234
        kfold_0 = cross_validation.KFold(n=_var91, n_folds=i, shuffle=_var92, random_state=_var93)
        _var94 = []
        _var95 = []
        test_acc_0 = _var94
        train_acc_0 = _var95
        for _var96 in kfold_0:
            _var99 = 0
            train_index_0 = _var96[_var99]
            _var100 = 1
            test_index_0 = _var96[_var100]
            _var101 = x_3[train_index_0]
            _var102 = y_2[train_index_0]
            clf_1 = classifier_4.fit(_var101, _var102)
            classifier_5 = clf_1
            _var103 = x_3[train_index_0]
            _var104 = y_2[train_index_0]
            _var105 = clf_1.score(_var103, _var104)
            train_acc_0.append(_var105)
            _var106 = x_3[test_index_0]
            _var107 = y_2[test_index_0]
            _var108 = clf_1.score(_var106, _var107)
            test_acc_0.append(_var108)
        classifier_6 = __phi__(classifier_5, classifier_4)
        _var109 = np.array(train_acc_0)
        _var110 = _var109.mean()
        k_train_acc.append(_var110)
        _var111 = np.array(test_acc_0)
        _var112 = _var111.mean()
        k_test_acc.append(_var112)
    classifier_7 = __phi__(classifier_6, classifier_4)
    plt.figure()
    _var113 = 2
    _var114 = range(_var113, max_num_k)
    _var115 = list(_var114)
    plt.plot(_var115, k_train_acc)
    _var116 = 2
    _var117 = range(_var116, max_num_k)
    _var118 = list(_var117)
    plt.plot(_var118, k_test_acc)
    return clf_1
_var119 = naive_bayes.MultinomialNB()
_var120 = 20
clf_2 = find_k(_var119, x_ones, y, _var120)
from sklearn.metrics import confusion_matrix
y_true = y
y_pred = clf_2.predict(x_0)
'\nNote! the confusion matrix here will be [0 1],\nnot [1, 0] as in the above image.\n'
conf = confusion_matrix(y_true, y_pred)
print(conf)
_var121 = clf_2.score(x_0, y)
print(_var121)
_var122 = 0
_var123 = 0
_var124 = (_var122, _var123)
_var125 = conf[_var124]
_var126 = 0
_var127 = 0
_var128 = (_var126, _var127)
_var129 = conf[_var128]
_var130 = 0
_var131 = 1
_var132 = (_var130, _var131)
_var133 = conf[_var132]
_var134 = (_var129 + _var133)
_var135 = (_var125 / _var134)
print(_var135)
_var136 = 1
_var137 = 1
_var138 = (_var136, _var137)
_var139 = conf[_var138]
_var140 = 1
_var141 = 0
_var142 = (_var140, _var141)
_var143 = conf[_var142]
_var144 = 1
_var145 = 1
_var146 = (_var144, _var145)
_var147 = conf[_var146]
_var148 = (_var143 + _var147)
_var149 = (_var139 / _var148)
print(_var149)
_var150 = clf_2.predict_proba(x_0)
_var151 = 0
prob = _var150[:, _var151]
_var152 = 0
_var153 = (y == _var152)
_var154 = prob[_var153]
_var155 = np.argsort(_var154)
_var156 = 5
bad_rotten = _var155[:_var156]
_var157 = 1
_var158 = (y == _var157)
_var159 = prob[_var158]
_var160 = np.argsort(_var159)
_var161 = (- 5)
bad_fresh = _var160[_var161:]
_var162 = 'Mis-predicted Rotten quotes'
print(_var162)
_var163 = '---------------------------'
print(_var163)
for row in bad_rotten:
    _var164 = 0
    _var165 = (y == _var164)
    _var166 = critics[_var165]
    _var167 = _var166.quote
    _var168 = _var167.irow(row)
    print(_var168)
    print()
_var169 = 'Mis-predicted Fresh quotes'
print(_var169)
_var170 = '--------------------------'
print(_var170)
for row_0 in bad_fresh:
    _var171 = 1
    _var172 = (y == _var171)
    _var173 = critics[_var172]
    _var174 = _var173.quote
    _var175 = _var174.irow(row_0)
    print(_var175)
    print()
from sklearn.feature_selection import f_classif
_var176 = get_ipython()
_var177 = 'pinfo'
_var178 = 'f_classif'
_var176.run_line_magic(_var177, _var178)
_var179 = f_classif(x_0, y)
print(_var179)
_var180 = f_classif(x_0, y)
_var181 = len(_var180)
print(_var181)
_var182 = f_classif(x_0, y)
_var183 = 0
_var184 = _var182[_var183]
_var185 = len(_var184)
print(_var185)
_var186 = f_classif(x_0, y)
_var187 = 1
_var188 = _var186[_var187]
_var189 = len(_var188)
print(_var189)
_var190 = 8
_var191 = 6
_var192 = (_var190, _var191)
_var193 = 80
plt.figure(figsize=_var192, dpi=_var193)
_var194 = f_classif(x_0, y)
_var195 = 0
_var196 = _var194[_var195]
_var197 = np.sort(_var196)
_var198 = 'b'
ax1 = plt.plot(_var197, color=_var198)
_var199 = 'F-value'
plt.ylabel(_var199)
ax2 = plt.twinx()
_var200 = f_classif(x_0, y)
_var201 = 1
_var202 = _var200[_var201]
_var203 = np.sort(_var202)
_var204 = 'g'
ax2.plot(_var203, color=_var204)
_var205 = 'p-value'
plt.ylabel(_var205)
_var206 = 90
_var207 = 100
_var208 = 1
_var209 = range(_var206, _var207, _var208)
for i_0 in _var209:
    _var210 = f_classif(x_0, y)
    _var211 = 0
    _var212 = _var210[_var211]
    _var213 = np.percentile(_var212, i_0)
    _var214 = f_classif(x_0, y)
    _var215 = 1
    _var216 = _var214[_var215]
    _var217 = np.percentile(_var216, i_0)
    _var218 = (i_0, _var213, _var217)
    print(_var218)
_var219 = f_classif(x_0, y)
_var220 = 0
_var221 = _var219[_var220]
_var222 = f_classif(x_0, y)
_var223 = 0
_var224 = _var222[_var223]
_var225 = 95
_var226 = np.percentile(_var224, _var225)
mask = (_var221 >= _var226)
new_features = x_0[:, mask]
np.shape(new_features)
_var227 = naive_bayes.MultinomialNB()
_var228 = 0.2
train_and_measure(_var227, new_features, y, _var228)

def train_and_measure(classifier_8, x_4, y_3, test_size_0):
    'modifying the function from class to return the training and test scores'
    from sklearn import cross_validation
    _var233 = 0.2
    _var234 = 1234
    (_var229, _var230, _var231, _var232) = cross_validation.train_test_split(x_4, y_3, test_size=_var233, random_state=_var234)
    xtrain_0 = _var229
    xtest_0 = _var230
    ytrain_0 = _var231
    ytest_0 = _var232
    clf_3 = classifier_8.fit(xtrain_0, ytrain_0)
    classifier_9 = clf_3
    training_accuracy_0 = clf_3.score(xtrain_0, ytrain_0)
    test_accuracy_0 = clf_3.score(xtest_0, ytest_0)
    return (training_accuracy_0, test_accuracy_0)
percentile = []
feature_percentage = []
train_acc_1 = []
test_acc_1 = []
_var235 = 900
_var236 = 1000
_var237 = 2
_var238 = range(_var235, _var236, _var237)
for i_1 in _var238:
    _var239 = f_classif(x_0, y)
    _var240 = 0
    _var241 = _var239[_var240]
    _var242 = f_classif(x_0, y)
    _var243 = 0
    _var244 = _var242[_var243]
    _var245 = 10
    _var246 = (i_1 / _var245)
    _var247 = np.percentile(_var244, _var246)
    mask_0 = (_var241 >= _var247)
    new_features_0 = x_0[:, mask_0]
    _var248 = np.shape(new_features_0)
    _var249 = 1
    _var250 = _var248[_var249]
    _var251 = np.shape(x_0)
    _var252 = 1
    _var253 = _var251[_var252]
    ratio = (_var250 / _var253)
    np.shape(new_features_0)
    _var254 = 10
    _var255 = (i_1 / _var254)
    percentile.append(_var255)
    feature_percentage.append(ratio)
    _var256 = naive_bayes.MultinomialNB()
    _var257 = 0.2
    scores = train_and_measure(_var256, new_features_0, y, _var257)
    _var258 = 0
    _var259 = scores[_var258]
    train_acc_1.append(_var259)
    _var260 = 1
    _var261 = scores[_var260]
    test_acc_1.append(_var261)
mask_1 = __phi__(mask_0, mask)
new_features_1 = __phi__(new_features_0, new_features)
_var262 = np.array(percentile)
_var263 = np.array(feature_percentage)
_var264 = np.array(train_acc_1)
_var265 = np.array(test_acc_1)
_var266 = {'percentile': _var262, 'feature_percentage': _var263, 'train_acc': _var264, 'test_acc': _var265}
df = pd.DataFrame(_var266)
_var267 = 8
_var268 = 6
_var269 = (_var267, _var268)
_var270 = 80
plt.figure(figsize=_var269, dpi=_var270)
_var271 = df.percentile
_var272 = df.train_acc
_var273 = 'Training Accuracy'
plt.plot(_var271, _var272, label=_var273)
_var274 = df.percentile
_var275 = df.test_acc
_var276 = 'Test Accuracy'
plt.plot(_var274, _var275, label=_var276)
_var277 = [0.6, 1]
plt.ylim(_var277)
_var278 = 'Percentile for F-statistic'
plt.xlabel(_var278)
_var279 = 'Accuracy'
plt.ylabel(_var279)
plt.legend()
ax1_0 = plt.twinx()
_var280 = df.percentile
_var281 = df.feature_percentage
_var282 = 0.2
_var283 = 0.2
ax1_0.bar(_var280, _var281, width=_var282, alpha=_var283)
_var284 = [90, 100]
plt.xlim(_var284)
_var285 = 'Ratio of feautures used vs all features model'
plt.ylabel(_var285)
plt.show()

def ad_words(text_0):
    import nltk
    token = nltk.word_tokenize(text_0)
    tagger = nltk.pos_tag(token)
    bag_of_words = [j[0] for j in tagger if (j[1] in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'))]
    _var286 = ' '
    _var287 = set(bag_of_words)
    _var288 = list(_var287)
    _var289 = _var286.join(_var288)
    _var290 = ''
    _var291 = (_var289 if bag_of_words else _var290)
    return _var291
_var292 = 'pos'
_var293 = critics.quote
_var294 = _var293.apply(ad_words)
critics_0 = set_index_wrapper(critics, _var292, _var294)
_var295 = 'pos'
_var296 = critics_0[_var295]
_var297 = 20
_var296.head(_var297)
from sklearn.feature_extraction.text import CountVectorizer
_var298 = 1
_var299 = 1
_var300 = (_var298, _var299)
vectorizer_2 = CountVectorizer(ngram_range=_var300)
_var301 = critics_0.pos
x_5 = vectorizer_2.fit_transform(_var301)
_var302 = np.shape(x_5)
_var303 = 1
_var302[_var303]
_var304 = naive_bayes.MultinomialNB()
_var305 = 0.2
train_and_measure(_var304, x_5, y, test_size_0=_var305)
_var306 = critics_0.pos
_var307 = ''
_var308 = (_var306 == _var307)
critics_0[_var308]
