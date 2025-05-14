

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
    _var10 = 'OverlapLeak.csv'
    test_fact_path = _var4.join(_var5, _var6, _var9, _var10)
    main.main(test_file_path)
    _var11 = os.path
    _var12 = _var11.exists(test_fact_path)
    _var13 = 'Leak result not found!'
    assert _var12, _var13
    _var14 = '\t'
    _var15 = ['model1', 'ctx1', 'model2', 'ctx2', 'invo', 'method']
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
    _var53 = 'testOversampler.py'
    _var54 = ['HistGradientBoostingClassifier.fit']
    template(_var53, _var54)
    _var55 = 'testOversampler2.py'
    _var56 = []
    template(_var55, _var56)

def test_smote():
    _var57 = 'yogmoh_news-category.py'
    _var58 = ['RandomForestClassifier.fit', 'Unknown.fit', 'XGBClassifier.fit']
    template(_var57, _var58)

def test_advanced():
    _var59 = 'dktalaicha_credit-card-fraud-detection-using-smote-adasyn.py'
    _var60 = ['LogisticRegression.fit', 'Unknown.fit']
    template(_var59, _var60)
    _var61 = 'nb_598984.py'
    _var62 = []
    template(_var61, _var62)

def test_funcdef():
    _var63 = 'shahules_tackling-class-imbalance.py'
    _var64 = ['LogisticRegression.fit']
    template(_var63, _var64)
