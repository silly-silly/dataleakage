

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
import os
_var0 = '/kaggle/input'
_var1 = os.walk(_var0)
for _var2 in _var1:
    _var6 = 0
    dirname = _var2[_var6]
    _var7 = 1
    _ = _var2[_var7]
    _var8 = 2
    filenames = _var2[_var8]
    for filename in filenames:
        _var9 = os.path
        _var10 = _var9.join(dirname, filename)
        print(_var10)
_var11 = get_ipython()
_var12 = 'pip install -U texthero'
_var11.system(_var12)
import tensorflow as tf
import pandas as pd
import numpy as np
import texthero as hero
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
_var13 = mpl.rcParams
_var14 = 'figure.dpi'
_var15 = 300
_var13_0 = set_index_wrapper(_var13, _var14, _var15)
_var16 = '/kaggle/input/new-category/Data_Train.xlsx'
train = pd.read_excel(_var16)
_var17 = '/kaggle/input/new-category/Data_Test.xlsx'
test = pd.read_excel(_var17)
_var18 = 5
_var19 = train.sample(_var18)
display(_var19)
_var20 = train.info()
display(_var20)
_var21 = test.info()
display(_var21)
_var22 = train.STORY
_var23 = 0
_var22[_var23]
_var24 = train.SECTION
_var25 = True
_var24.value_counts(normalize=_var25)
_var26 = 'SECTION'
_var27 = 1
_var28 = train.drop(_var26, axis=_var27)
_var29 = [_var28, test]
combined_df = pd.concat(_var29)
combined_df.info()
_var30 = hero.visualization
_var31 = 'STORY'
_var32 = combined_df[_var31]
_var33 = 1000
_var34 = 'BLACK'
_var30.wordcloud(_var32, max_words=_var33, background_color=_var34)
hero.Word2Vec
_var35 = 'cleaned_text'
_var36 = 'STORY'
_var37 = combined_df[_var36]
_var38 = hero.remove_angle_brackets
_var39 = _var37.pipe(_var38)
_var40 = hero.remove_brackets
_var41 = _var39.pipe(_var40)
_var42 = hero.remove_curly_brackets
_var43 = _var41.pipe(_var42)
_var44 = hero.remove_diacritics
_var45 = _var43.pipe(_var44)
_var46 = hero.remove_digits
_var47 = _var45.pipe(_var46)
_var48 = hero.remove_html_tags
_var49 = _var47.pipe(_var48)
_var50 = hero.remove_punctuation
_var51 = _var49.pipe(_var50)
_var52 = hero.remove_round_brackets
_var53 = _var51.pipe(_var52)
_var54 = hero.remove_square_brackets
_var55 = _var53.pipe(_var54)
_var56 = hero.remove_stopwords
_var57 = _var55.pipe(_var56)
_var58 = hero.remove_urls
_var59 = _var57.pipe(_var58)
_var60 = hero.remove_whitespace
_var61 = _var59.pipe(_var60)
_var62 = hero.lowercase
_var63 = _var61.pipe(_var62)
combined_df_0 = set_index_wrapper(combined_df, _var35, _var63)
lemm = WordNetLemmatizer()

def word_lemma(text):
    words = nltk.word_tokenize(text)
    lemma = [lemm.lemmatize(word) for word in words]
    _var64 = ' '
    joined_text = _var64.join(lemma)
    return joined_text
_var65 = 'lemmatized_text'
_var66 = combined_df_0.cleaned_text

def _func0(x):
    _var67 = word_lemma(x)
    return _var67
_var68 = _var66.apply(_func0)
combined_df_1 = set_index_wrapper(combined_df_0, _var65, _var68)
text_0 = []
_var69 = len(combined_df_1)
_var70 = range(_var69)
for i in _var70:
    _var71 = 'lemmatized_text'
    _var72 = combined_df_1[_var71]
    _var73 = _var72.iloc
    _var74 = _var73[i]
    review = nltk.word_tokenize(_var74)
    _var75 = ' '
    review_0 = _var75.join(review)
    text_0.append(review_0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
_var76 = 'lemmatized_text'
_var77 = combined_df_1[_var76]
_var78 = _var77.iloc
_var79 = 7628
X = _var78[:_var79]
_var80 = 'lemmatized_text'
_var81 = combined_df_1[_var80]
_var82 = _var81.iloc
_var83 = 7628
test_df = _var82[_var83:]
_var84 = 9000
cv = CountVectorizer(max_features=_var84)
cv_0 = cv.fit(X)
X_0 = cv_0.transform(X)
test_df_0 = cv_0.transform(test_df)
y = train.SECTION
_var85 = 9000
tfid = TfidfVectorizer(max_features=_var85)
tfid_0 = tfid.fit(X_0)
X_1 = tfid_0.transform(X_0)
test_df_1 = tfid_0.transform(test_df_0)
y_0 = train.SECTION
from imblearn.over_sampling import SMOTE
smote = SMOTE()
(_var86, _var87) = smote.fit_resample(X_1, y_0)
X_new = _var86
y_new = _var87
_var92 = 0.2
_var93 = 42
(_var88, _var89, _var90, _var91) = train_test_split(X_new, y_new, test_size=_var92, random_state=_var93, stratify=y_new)
X_train = _var88
X_test = _var89
y_train = _var90
y_test = _var91
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_0 = rf.fit(X_train, y_train)
y_pred = rf_0.predict(X_test)
accuracy_score(y_test, y_pred)
from xgboost import XGBClassifier
xg = XGBClassifier()
xg_0 = xg.fit(X_train, y_train)
y_pred_0 = xg_0.predict(X_test)
accuracy_score(y_test, y_pred_0)
from catboost import CatBoostClassifier
_var94 = 'GPU'
cat = CatBoostClassifier(task_type=_var94)
cat_0 = cat.fit(X_train, y_train)
y_pred_1 = cat_0.predict(X_test)
accuracy_score(y_test, y_pred_1)
predictions = xg_0.predict(test_df_1)
_var95 = {'SECTION': predictions}
submissions = pd.DataFrame(_var95)
_var96 = './sub8.csv'
_var97 = False
_var98 = True
submissions.to_csv(_var96, index=_var97, header=_var98)
