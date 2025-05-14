

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
import os, sys
import pandas as pd
from src import main

def template(test_file, taintMethods):
    _var0 = os.path
    _var1 = '.'
    _var2 = 'tests'
    _var3 = 'inputs'
    test_file_path = _var0.join(_var1, _var2, _var3, test_file)
    _var4 = os.path
    _var5 = 'tests'
    _var6 = 'inputs'
    _var7 = '.py'
    _var8 = '-fact'
    _var9 = test_file.replace(_var7, _var8)
    _var10 = 'PreProcessingLeak.csv'
    test_fact_path = _var4.join(_var5, _var6, _var9, _var10)
    main.main(test_file_path)
    _var11 = os.path
    _var12 = _var11.exists(test_fact_path)
    _var13 = 'Leak result not found!'
    assert _var12, _var13
    _var14 = '\t'
    _var15 = ['heap', 'invo', 'method', 'ctx']
    df = pd.read_csv(test_fact_path, sep=_var14, names=_var15)
    _var16 = 'method'
    _var17 = df[_var16]
    print(_var17)

    def report():
        _var18 = global_wrapper(taintMethods)
        _var19 = set(_var18)
        _var20 = global_wrapper(df)
        _var21 = 'method'
        _var22 = _var20[_var21]
        _var23 = set(_var22)
        _var24 = _var19.difference(_var23)
        print(_var24)
        _var25 = global_wrapper(taintMethods)
        _var26 = set(_var25)
        _var27 = global_wrapper(df)
        _var28 = 'method'
        _var29 = _var27[_var28]
        _var30 = set(_var29)
        _var31 = _var26.difference(_var30)
        _var32 = len(_var31)
        _var33 = 0
        hasFalseNeg = (_var32 > _var33)
        _var34 = global_wrapper(df)
        _var35 = 'method'
        _var36 = _var34[_var35]
        _var37 = set(_var36)
        _var38 = global_wrapper(taintMethods)
        _var39 = set(_var38)
        _var40 = _var37.difference(_var39)
        _var41 = len(_var40)
        _var42 = 0
        hasFalsePos = (_var41 > _var42)
        if hasFalseNeg:
            _var43 = 'Leak undetected!!!'
            return _var43
        if hasFalsePos:
            _var44 = 'False leak detected!!!'
            return _var44
        _var45 = False
        _var46 = 'Should not reach here'
        assert _var45, _var46
    _var47 = 'method'
    _var48 = df[_var47]
    _var49 = set(_var48)
    _var50 = set(taintMethods)
    _var51 = (_var49 == _var50)
    _var52 = report()
    assert _var51, _var52

def test_basic():
    _var53 = 'test0.py'
    _var54 = ['LogisticRegression.fit']
    template(_var53, _var54)
    _var55 = 'test1.py'
    _var56 = ['LogisticRegression.fit']
    template(_var55, _var56)
    _var57 = 'test2.py'
    _var58 = ['LogisticRegression.fit']
    template(_var57, _var58)
    _var59 = 'test3.py'
    _var60 = []
    template(_var59, _var60)
    _var61 = 'test4.py'
    _var62 = []
    template(_var61, _var62)

def test_fillna():
    _var63 = 'titanic0.py'
    _var64 = ['LogisticRegression.fit']
    template(_var63, _var64)
    _var65 = 'nb_344873.py'
    _var66 = ['Sequential.fit', 'LassoCV.fit', 'XGBRegressor.fit']
    template(_var65, _var66)

def test_tfidf():
    _var67 = 'nb_100841.py'
    _var68 = ['MultinomialNB.fit']
    template(_var67, _var68)
    _var69 = 'nb_334422.py'
    _var70 = ['AdaBoostRegressor.fit', 'DecisionTreeRegressor.fit', 'GradientBoostingRegressor.fit', 'KNeighborsRegressor.fit', 'LassoCV.fit', 'LinearRegression.fit', 'RandomForestRegressor.fit', 'RidgeCV.fit']
    template(_var69, _var70)

def test_dataFrameMapper():
    _var71 = 'nb_132929.py'
    _var72 = ['GaussianNB.fit', 'SGDClassifier.fit']
    template(_var71, _var72)

def test_scaler():
    _var73 = 'nb_194503.py'
    _var74 = ['Model.fit']
    template(_var73, _var74)
    _var75 = 'nb_362989.py'
    _var76 = ['GaussianNB.fit', 'SGDClassifier.fit']
    template(_var75, _var76)
    _var77 = 'nb_473437.py'
    _var78 = []
    template(_var77, _var78)

def test_pca():
    _var79 = 'nb_205857.py'
    _var80 = ['RandomForestClassifier.fit', 'Unknown.fit']
    template(_var79, _var80)
    _var81 = 'nb_471253.py'
    _var82 = ['SGDClassifier.fit']
    template(_var81, _var82)

def test_countvec():
    _var83 = 'nb_303674.py'
    _var84 = ['Unknown.fit']
    template(_var83, _var84)
    _var85 = 'nb_1020535.py'
    _var86 = []
    template(_var85, _var86)

def test_pipeline():
    _var87 = 'nb_276778.py'
    _var88 = ['RandomizedSearchCV.fit']
    template(_var87, _var88)
    _var89 = 'nb_277256.py'
    _var90 = []
    template(_var89, _var90)

def test_feature_selection():
    _var91 = 'nb_387986.py'
    _var92 = ['RandomForestRegressor.fit', 'LinearRegression.fit', 'Pipeline.fit', 'RidgeCV.fit']
    template(_var91, _var92)

def test_applymap():
    _var93 = 'nb_344814.py'
    _var94 = ['LogisticRegression.fit']
    template(_var93, _var94)

def test_equiv_edge():
    _var95 = 'nb_282393.py'
    _var96 = ['Sequential.fit_generator']
    template(_var95, _var96)

def test_loop():
    _var97 = 'nb_175471.py'
    _var98 = ['MultinomialNB.fit', 'Unknown.fit']
    template(_var97, _var98)
    _var99 = 'nb_248151.py'
    _var100 = ['Unknown.fit']
    template(_var99, _var100)

def test_context_sensitivity():
    _var101 = 'nb_273933.py'
    _var102 = ['GridSearchCV.fit', 'Unknown.fit', 'KNeighborsClassifier.fit']
    template(_var101, _var102)

def test_classdef():
    _var103 = 'nb_424904.py'
    _var104 = ['Unknown.fit']
    template(_var103, _var104)

def test_funcdef():
    _var105 = 'nb_292583.py'
    _var106 = ['AdaBoostClassifier.fit', 'Any | Unknown | type.fit', 'Unknown.fit']
    template(_var105, _var106)
    _var107 = 'nb_481597.py'
    _var108 = ['RandomForestRegressor.fit']
    template(_var107, _var108)

def test_branch():
    _var109 = 'nb_1080148.py'
    _var110 = ['Unknown.fit']
    template(_var109, _var110)

def test_cut():
    _var111 = 'nb_1090319.py'
    _var112 = ['Unknown.fit', 'LogisticRegression.fit']
    template(_var111, _var112)
    _var113 = 'nb_1255091.py'
    _var114 = ['DecisionTreeClassifier.fit', 'KNeighborsClassifier.fit', 'RandomForestClassifier.fit', 'RandomForestRegressor | ExtraTreesRegressor | BaseForest.fit']
    template(_var113, _var114)

def test_torch():
    _var115 = 'nb_504.py'
    _var116 = ['model_0']
    template(_var115, _var116)
