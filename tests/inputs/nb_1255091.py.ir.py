

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
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import LogFormatter
_var0 = get_ipython()
_var1 = 'matplotlib'
_var2 = 'inline'
_var0.run_line_magic(_var1, _var2)
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
_var3 = get_ipython()
_var4 = 'matplotlib'
_var5 = 'inline'
_var3.run_line_magic(_var4, _var5)
_var6 = 'ListofInformalSettlements_29_DEC_2016_All_sites.csv'
df = pd.read_csv(_var6)
df_base = df
df_base.head()
_var7 = 'Governorate'
_var8 = df[_var7]
_var8.unique()
import urllib.request, urllib.parse, urllib.error, json, time
LocDict = dict()
_var9 = '../../PythonLearningSideProjects/APIs/Sarine_APIs.csv'
api_table = pd.read_csv(_var9)
_var10 = 'API key'
_var11 = api_table[_var10]
_var12 = 'Name'
_var13 = api_table[_var12]
_var14 = ['google_api']
_var15 = _var13.isin(_var14)
_var16 = _var11[_var15]
_var17 = _var16.values
_var18 = 0
google_api = _var17[_var18]
_var19 = 'Governorate'
_var20 = df_base[_var19]
labeltext = _var20.unique()
_var21 = len(labeltext)
_var22 = range(_var21)
for ii in _var22:
    _var23 = ''
    _var24 = labeltext[ii]
    _var25 = (_var23 + _var24)
    _var26 = ' Governorate, Lebanon'
    address = (_var25 + _var26)
    _var27 = 'address='
    _var28 = ' '
    _var29 = '+'
    _var30 = address.replace(_var28, _var29)
    addP = (_var27 + _var30)
    _var31 = 'https://maps.googleapis.com/maps/api/geocode/json?'
    _var32 = (_var31 + addP)
    _var33 = '&key='
    _var34 = (_var32 + _var33)
    GeoUrl = (_var34 + google_api)
    _var35 = global_wrapper(urllib)
    _var36 = _var35.request
    response = _var36.urlopen(GeoUrl)
    jsonRaw = response.read()
    jsonData = json.loads(jsonRaw)
    _var37 = 'status'
    _var38 = jsonData[_var37]
    _var39 = 'OK'
    _var40 = (_var38 == _var39)
    if _var40:
        _var41 = 'results'
        _var42 = jsonData[_var41]
        _var43 = 0
        res = _var42[_var43]
        _var44 = labeltext[ii]
        _var45 = 'geometry'
        _var46 = res[_var45]
        _var47 = 'location'
        _var48 = _var46[_var47]
        _var49 = 'lng'
        _var50 = _var48[_var49]
        _var51 = 'geometry'
        _var52 = res[_var51]
        _var53 = 'location'
        _var54 = _var52[_var53]
        _var55 = 'lat'
        _var56 = _var54[_var55]
        _var57 = [_var50, _var56]
        LocDict_0 = set_index_wrapper(LocDict, _var44, _var57)
    else:
        LocDict_1 = {None, None, None}
    LocDict_2 = __phi__(LocDict_0, LocDict_1)
LocDict_3 = __phi__(LocDict_2, LocDict)
print(LocDict_3)
uppercorner = [34.7, 36.8]
lowercorner = [33.018977, 34.9]
_var58 = (- 50)
_var59 = 5
_var60 = (_var58, _var59)
_var61 = (- 120)
_var62 = 0
_var63 = (_var61, _var62)
_var64 = (- 50)
_var65 = 5
_var66 = (_var64, _var65)
_var67 = (- 50)
_var68 = 5
_var69 = (_var67, _var68)
_var70 = (- 50)
_var71 = 5
_var72 = (_var70, _var71)
_var73 = (- 50)
_var74 = 10
_var75 = (_var73, _var74)
xytexts = [_var60, _var63, _var66, _var69, _var72, _var75]
_var76 = 14
_var77 = 10
_var78 = (_var76, _var77)
fig = plt.figure(figsize=_var78)
_var79 = 111
ax = fig.add_subplot(_var79)
_var80 = 'merc'
_var81 = 1
_var82 = lowercorner[_var81]
_var83 = 0
_var84 = lowercorner[_var83]
_var85 = 1
_var86 = uppercorner[_var85]
_var87 = 0
_var88 = uppercorner[_var87]
_var89 = 0
_var90 = lowercorner[_var89]
_var91 = 'i'
_var92 = 22770
m = Basemap(projection=_var80, llcrnrlon=_var82, llcrnrlat=_var84, urcrnrlon=_var86, urcrnrlat=_var88, lat_ts=_var90, resolution=_var91, epsg=_var92)
_var93 = 'ESRI_Imagery_World_2D'
_var94 = 300
_var95 = True
m.arcgisimage(service=_var93, xpixels=_var94, verbose=_var95)
_var96 = 'darkcyan'
_var97 = 1.5
_var98 = 1
m.drawrivers(color=_var96, linewidth=_var97, zorder=_var98)
_var99 = 'Number of Ind'
_var100 = df[_var99]
_var101 = _var100.values
_var102 = 'Number of Ind in SSBs'
_var103 = df[_var102]
_var104 = _var103.values
plotvar = (_var101 + _var104)
_var107 = 'Longitude'
_var108 = df[_var107]
_var109 = _var108.values
_var110 = 'Latitude'
_var111 = df[_var110]
_var112 = _var111.values
(_var105, _var106) = m(_var109, _var112)
x = _var105
y = _var106
_var113 = len(labeltext)
_var114 = range(_var113)
for ii_0 in _var114:
    _var117 = global_wrapper(labelcoord)
    _var118 = _var117[ii_0]
    _var119 = 0
    _var120 = _var118[_var119]
    _var121 = global_wrapper(labelcoord)
    _var122 = _var121[ii_0]
    _var123 = 1
    _var124 = _var122[_var123]
    (_var115, _var116) = m(_var120, _var124)
    xlab = _var115
    ylab = _var116
    _var127 = labeltext[ii_0]
    _var128 = LocDict_3[_var127]
    _var129 = 0
    _var130 = _var128[_var129]
    _var131 = labeltext[ii_0]
    _var132 = LocDict_3[_var131]
    _var133 = 1
    _var134 = _var132[_var133]
    (_var125, _var126) = m(_var130, _var134)
    xlab_0 = _var125
    ylab_0 = _var126
    _var135 = 'lightgray'
    _var136 = '*'
    _var137 = 15
    _var138 = 2
    m.plot(xlab_0, ylab_0, color=_var135, marker=_var136, markersize=_var137, zorder=_var138)
    _var139 = labeltext[ii_0]
    _var140 = (xlab_0, ylab_0)
    _var141 = xytexts[ii_0]
    _var142 = 'offset points'
    _var143 = 16
    _var144 = 'lightgray'
    plt.annotate(_var139, xy=_var140, xytext=_var141, textcoords=_var142, fontsize=_var143, color=_var144)
_var145 = 100
_var146 = 'log'
_var147 = 1
_var148 = cm.YlOrRd_r
_var149 = 2
m.hexbin(x, y, gridsize=_var145, bins=_var146, mincnt=_var147, cmap=_var148, zorder=_var149)
_var150 = 10
_var151 = False
formatter = LogFormatter(_var150, labelOnlyBase=_var151)
cb = plt.colorbar()
_var152 = 'Number of  settlements [log10(N)]'
cb.set_label(_var152)
_var153 = '/Users/Sarine/Documents/Sarine/Other/PythonLearningSideProjects/lebanon-refugee-data/LBN_adm/LBN_adm1'
_var154 = 'areas'
_var155 = 2
_var156 = 1
m.readshapefile(_var153, _var154, linewidth=_var155, zorder=_var156)
_var157 = 20
_var158 = 10
_var159 = (_var157, _var158)
fig_0 = plt.figure(figsize=_var159)
_var160 = 121
ax_0 = fig_0.add_subplot(_var160)
_var161 = 'merc'
_var162 = 1
_var163 = lowercorner[_var162]
_var164 = 0
_var165 = lowercorner[_var164]
_var166 = 1
_var167 = uppercorner[_var166]
_var168 = 0
_var169 = uppercorner[_var168]
_var170 = 0
_var171 = lowercorner[_var170]
_var172 = 'i'
_var173 = 22770
m_0 = Basemap(projection=_var161, llcrnrlon=_var163, llcrnrlat=_var165, urcrnrlon=_var167, urcrnrlat=_var169, lat_ts=_var171, resolution=_var172, epsg=_var173)
_var174 = 'ESRI_Imagery_World_2D'
_var175 = 300
_var176 = True
m_0.arcgisimage(service=_var174, xpixels=_var175, verbose=_var176)
_var177 = 2
m_0.drawcoastlines(linewidth=_var177)
_var178 = 2
_var179 = 1
m_0.drawcountries(linewidth=_var178, zorder=_var179)
_var180 = 'Number of Ind'
_var181 = df[_var180]
_var182 = _var181.values
_var183 = 'Number of Ind in SSBs'
_var184 = df[_var183]
_var185 = _var184.values
plotvar_0 = (_var182 + _var185)
_var188 = 'Longitude'
_var189 = df[_var188]
_var190 = _var189.values
_var191 = 'Latitude'
_var192 = df[_var191]
_var193 = _var192.values
(_var186, _var187) = m_0(_var190, _var193)
x_0 = _var186
y_0 = _var187
_var194 = np.mean
_var195 = 100
_var196 = 1
_var197 = cm.YlOrRd_r
_var198 = 2
_var199 = global_wrapper(matplotlib)
_var200 = _var199.colors
_var201 = _var200.LogNorm()
m_0.hexbin(x_0, y_0, C=plotvar_0, reduce_C_function=_var194, gridsize=_var195, mincnt=_var196, cmap=_var197, zorder=_var198, norm=_var201)
_var202 = 10
_var203 = False
formatter_0 = LogFormatter(_var202, labelOnlyBase=_var203)
cb_0 = plt.colorbar(format=formatter_0)
_var204 = [1, 10, 50, 100, 500]
_var205 = True
cb_0.set_ticks(_var204, update_ticks=_var205)
_var206 = ['1', '10', '50', '100', '500']
_var207 = True
cb_0.set_ticklabels(_var206, update_ticks=_var207)
_var208 = 'Mean number of people in settlements'
cb_0.set_label(_var208)
_var209 = 122
ax1 = fig_0.add_subplot(_var209)
_var210 = 'merc'
_var211 = 1
_var212 = lowercorner[_var211]
_var213 = 0
_var214 = lowercorner[_var213]
_var215 = 1
_var216 = uppercorner[_var215]
_var217 = 0
_var218 = uppercorner[_var217]
_var219 = 0
_var220 = lowercorner[_var219]
_var221 = 'i'
_var222 = 22770
m1 = Basemap(projection=_var210, llcrnrlon=_var212, llcrnrlat=_var214, urcrnrlon=_var216, urcrnrlat=_var218, lat_ts=_var220, resolution=_var221, epsg=_var222)
_var223 = 'ESRI_Imagery_World_2D'
_var224 = 300
_var225 = True
m1.arcgisimage(service=_var223, xpixels=_var224, verbose=_var225)
_var226 = 2
m1.drawcoastlines(linewidth=_var226)
_var227 = 2
_var228 = 1
m1.drawcountries(linewidth=_var227, zorder=_var228)
_var229 = 'Water Capacity in L'
_var230 = df[_var229]
plotvar_1 = _var230.values
_var233 = 'Longitude'
_var234 = df[_var233]
_var235 = _var234.values
_var236 = 'Latitude'
_var237 = df[_var236]
_var238 = _var237.values
(_var231, _var232) = m1(_var235, _var238)
x_1 = _var231
y_1 = _var232
_var239 = np.mean
_var240 = 100
_var241 = 1
_var242 = cm.YlOrRd_r
_var243 = 2
_var244 = global_wrapper(matplotlib)
_var245 = _var244.colors
_var246 = _var245.LogNorm()
m1.hexbin(x_1, y_1, C=plotvar_1, reduce_C_function=_var239, gridsize=_var240, mincnt=_var241, cmap=_var242, zorder=_var243, norm=_var246)
_var247 = 10
_var248 = False
formatter_1 = LogFormatter(_var247, labelOnlyBase=_var248)
cb_1 = plt.colorbar()
_var249 = [1, 10, 100, 1000, 10000, 50000]
_var250 = True
cb_1.set_ticks(_var249, update_ticks=_var250)
_var251 = ['1', '10', '100', '1000', '10000', '50000']
_var252 = True
cb_1.set_ticklabels(_var251, update_ticks=_var252)
_var253 = 'Water capacity in L'
cb_1.set_label(_var253)
df.describe()
_var254 = 'Status'
_var255 = df[_var254]
_var256 = ['Inactive']
_var257 = _var255.isin(_var256)
_var258 = (~ _var257)
_var259 = df[_var258]
_var260 = True
df_0 = _var259.reset_index(drop=_var260)
_var261 = 'Status'
_var262 = df_0[_var261]
_var263 = ['Erroneous']
_var264 = _var262.isin(_var263)
_var265 = (~ _var264)
_var266 = df_0[_var265]
_var267 = True
df_1 = _var266.reset_index(drop=_var267)
_var268 = 'Status'
_var269 = df_1[_var268]
_var270 = ['Not Willing']
_var271 = _var269.isin(_var270)
_var272 = (~ _var271)
_var273 = df_1[_var272]
_var274 = True
df_2 = _var273.reset_index(drop=_var274)
_var275 = 'Status'
_var276 = 'Status'
_var277 = df_2[_var276]
_var278 = {'Active': 1, 'Less than 4': 0}
_var279 = _var277.map(_var278)
_var280 = _var279.astype(int)
df_3 = set_index_wrapper(df_2, _var275, _var280)
_var281 = 'Status'
_var282 = df_3[_var281]
_var282.value_counts()
_var283 = 'Type of Water Source'
_var284 = df_3[_var283]
_var285 = _var284.value_counts()
_var285.sum()
_var286 = 'Waste Disposal'
_var287 = df_3[_var286]
_var288 = _var287.value_counts()
_var288.sum()
_var289 = 'Waste Water Disposal'
_var290 = df_3[_var289]
_var291 = _var290.value_counts()
_var291.sum()
_var292 = 'Type of Contract'
_var293 = df_3[_var292]
_var294 = _var293.value_counts()
_var294.sum()
_var295 = 'Type of Internet Connection'
_var296 = df_3[_var295]
_var297 = _var296.value_counts()
_var297.sum()
_var298 = 0
_var299 = ['Type of Water Source']
_var300 = 'all'
df_4 = df_3.dropna(axis=_var298, subset=_var299, how=_var300)
_var301 = 0
_var302 = ['Waste Disposal']
_var303 = 'all'
df_5 = df_4.dropna(axis=_var301, subset=_var302, how=_var303)
_var304 = 0
_var305 = ['Waste Water Disposal']
_var306 = 'all'
df_6 = df_5.dropna(axis=_var304, subset=_var305, how=_var306)
_var307 = 0
_var308 = ['Type of Contract']
_var309 = 'all'
df_7 = df_6.dropna(axis=_var307, subset=_var308, how=_var309)
_var310 = 0
_var311 = ['Type of Internet Connection']
_var312 = 'all'
df_8 = df_7.dropna(axis=_var310, subset=_var311, how=_var312)
_var313 = True
df_9 = df_8.reset_index(drop=_var313)
_var314 = 'Status'
g = sns.FacetGrid(df_9, col=_var314)
_var315 = plt.hist
_var316 = 'Water Capacity in L'
_var317 = 50
g.map(_var315, _var316, bins=_var317)
_var318 = 'Status'
g_0 = sns.FacetGrid(df_9, col=_var318)
_var319 = plt.hist
_var320 = 'Number of Latrines'
_var321 = 10
g_0.map(_var319, _var320, bins=_var321)
_var322 = 'Status'
g_1 = sns.FacetGrid(df_9, col=_var322)
_var323 = plt.hist
_var324 = 'Number of SSBs'
_var325 = 50
g_1.map(_var323, _var324, bins=_var325)
_var326 = 'Status'
g_2 = sns.FacetGrid(df_9, col=_var326)
_var327 = plt.hist
_var328 = 'Number of Ind in SSBs'
_var329 = 50
g_2.map(_var327, _var328, bins=_var329)
_var330 = 'Waste Disposal'
_var331 = 'Waste Disposal'
_var332 = df_9[_var331]
_var333 = {'Municipality Collection': 1, 'Burn it': 2, 'Dump it outside the camp': 3, 'Burry it': 3}
_var334 = _var332.map(_var333)
_var335 = _var334.astype(int)
df_10 = set_index_wrapper(df_9, _var330, _var335)
df_10.head()
_var336 = 'Waste Water Disposal'
_var337 = df_10[_var336]
_var337.value_counts()
_var338 = 'Waste Water Disposal'
_var339 = 'Waste Water Disposal'
_var340 = df_10[_var339]
_var341 = ['Storm water channel', 'Septic tank', 'Municipality sewer network / treated', 'Irrigation canal']
_var342 = 'Rare'
_var343 = _var340.replace(_var341, _var342)
df_11 = set_index_wrapper(df_10, _var338, _var343)
_var344 = 'Waste Water Disposal'
_var345 = df_11[_var344]
_var345.value_counts()
_var346 = 'Waste Water Disposal'
_var347 = 'Waste Water Disposal'
_var348 = df_11[_var347]
_var349 = {'Direct discharge to environment': 1, 'Cesspit': 2, 'Open pit': 3, 'Holding tank': 4, 'Municipality sewer network / not treated': 5, 'Rare': 6}
_var350 = _var348.map(_var349)
_var351 = _var350.astype(int)
df_12 = set_index_wrapper(df_11, _var346, _var351)
_var352 = 'Waste Water Disposal'
_var353 = df_12[_var352]
_var353.value_counts()
_var354 = ['Governorate', 'Status']
_var355 = df_12[_var354]
_var356 = ['Governorate']
_var357 = False
_var358 = _var355.groupby(_var356, as_index=_var357)
_var359 = _var358.mean()
_var360 = 'Status'
_var361 = False
_var359.sort_values(by=_var360, ascending=_var361)
_var362 = 'Governorate'
_var363 = 'Governorate'
_var364 = df_12[_var363]
_var365 = {'Bekaa': 1, 'North': 2, 'South': 3, 'Mount Lebanon': 4, 'Nabatiye': 5, 'Beirut': 6}
_var366 = _var364.map(_var365)
_var367 = _var366.astype(int)
df_13 = set_index_wrapper(df_12, _var362, _var367)
_var368 = 'Governorate'
_var369 = df_13[_var368]
_var369.value_counts()
_var370 = ['District', 'Status']
_var371 = df_13[_var370]
_var372 = ['District']
_var373 = False
_var374 = _var371.groupby(_var372, as_index=_var373)
_var375 = _var374.mean()
_var376 = 'Status'
_var377 = False
_var375.sort_values(by=_var376, ascending=_var377)
_var378 = ['Waste Disposal', 'Status']
_var379 = df_13[_var378]
_var380 = ['Waste Disposal']
_var381 = False
_var382 = _var379.groupby(_var380, as_index=_var381)
_var383 = _var382.mean()
_var384 = 'Status'
_var385 = False
_var383.sort_values(by=_var384, ascending=_var385)
_var386 = ['Waste Water Disposal', 'Status']
_var387 = df_13[_var386]
_var388 = ['Waste Water Disposal']
_var389 = False
_var390 = _var387.groupby(_var388, as_index=_var389)
_var391 = _var390.mean()
_var392 = 'Status'
_var393 = False
_var391.sort_values(by=_var392, ascending=_var393)
_var394 = 'Type of Water Source'
_var395 = df_13[_var394]
_var395.value_counts()
_var396 = 'Type of Water Source'
_var397 = 'Type of Water Source'
_var398 = df_13[_var397]
_var399 = ['Spring', 'Well', 'Others', 'River']
_var400 = 'Rare'
_var401 = _var398.replace(_var399, _var400)
df_14 = set_index_wrapper(df_13, _var396, _var401)
_var402 = ['Type of Water Source', 'Status']
_var403 = df_14[_var402]
_var404 = ['Type of Water Source']
_var405 = False
_var406 = _var403.groupby(_var404, as_index=_var405)
_var407 = _var406.mean()
_var408 = 'Status'
_var409 = False
_var407.sort_values(by=_var408, ascending=_var409)
_var410 = 'Type of Water Source'
_var411 = 'Type of Water Source'
_var412 = df_14[_var411]
_var413 = {'Water Trucking': 1, 'Borehole': 2, 'Water Network': 3, 'Rare': 4}
_var414 = _var412.map(_var413)
_var415 = _var414.astype(int)
df_15 = set_index_wrapper(df_14, _var410, _var415)
_var416 = 'Type of Water Source'
_var417 = df_15[_var416]
_var417.value_counts()
_var418 = 'Type of Contract'
_var419 = df_15[_var418]
_var419.value_counts()
_var420 = ['Type of Contract', 'Status']
_var421 = df_15[_var420]
_var422 = ['Type of Contract']
_var423 = False
_var424 = _var421.groupby(_var422, as_index=_var423)
_var425 = _var424.mean()
_var426 = 'Status'
_var427 = False
_var425.sort_values(by=_var426, ascending=_var427)
_var428 = 'Type of Contract'
_var429 = 'Type of Contract'
_var430 = df_15[_var429]
_var431 = {'Verbal': 1, 'None': 2, 'Written': 3}
_var432 = _var430.map(_var431)
_var433 = _var432.astype(int)
df_16 = set_index_wrapper(df_15, _var428, _var433)
_var434 = 'Type of Contract'
_var435 = df_16[_var434]
_var435.value_counts()
_var436 = 'Type of Internet Connection'
_var437 = df_16[_var436]
_var437.value_counts()
_var438 = ['Type of Internet Connection', 'Status']
_var439 = df_16[_var438]
_var440 = ['Type of Internet Connection']
_var441 = False
_var442 = _var439.groupby(_var440, as_index=_var441)
_var443 = _var442.mean()
_var444 = 'Status'
_var445 = False
_var443.sort_values(by=_var444, ascending=_var445)
_var446 = 'Type of Internet Connection'
_var447 = 'Type of Internet Connection'
_var448 = df_16[_var447]
_var449 = {'Mobile network - 3G / 4G': 1, 'Wifi  / local Internet service provider': 2, 'No Internet access': 3}
_var450 = _var448.map(_var449)
_var451 = _var450.astype(int)
df_17 = set_index_wrapper(df_16, _var446, _var451)
_var452 = 'Type of Internet Connection'
_var453 = df_17[_var452]
_var453.value_counts()
_var454 = ['Consultation Fee for PHC (3000/5000)', 'Status']
_var455 = df_17[_var454]
_var456 = ['Consultation Fee for PHC (3000/5000)']
_var457 = False
_var458 = _var455.groupby(_var456, as_index=_var457)
_var459 = _var458.mean()
_var460 = 'Status'
_var461 = False
_var459.sort_values(by=_var460, ascending=_var461)
_var462 = ['Free Vaccination for Children under 12', 'Status']
_var463 = df_17[_var462]
_var464 = ['Free Vaccination for Children under 12']
_var465 = False
_var466 = _var463.groupby(_var464, as_index=_var465)
_var467 = _var466.mean()
_var468 = 'Status'
_var469 = False
_var467.sort_values(by=_var468, ascending=_var469)
_var470 = 'LatrinesBand'
_var471 = 'Number of Latrines'
_var472 = df_17[_var471]
_var473 = 4
_var474 = pd.qcut(_var472, _var473)
df_18 = set_index_wrapper(df_17, _var470, _var474)
_var475 = ['LatrinesBand', 'Status']
_var476 = df_18[_var475]
_var477 = ['LatrinesBand']
_var478 = False
_var479 = _var476.groupby(_var477, as_index=_var478)
_var480 = _var479.mean()
_var481 = 'LatrinesBand'
_var482 = True
_var480.sort_values(by=_var481, ascending=_var482)
_var483 = df_18.loc
_var484 = 'Number of Latrines'
_var485 = df_18[_var484]
_var486 = 1
_var487 = (_var485 <= _var486)
_var488 = 'LatrinesBandNum'
_var489 = (_var487, _var488)
_var490 = 0
_var483_0 = set_index_wrapper(_var483, _var489, _var490)
_var491 = df_18.loc
_var492 = 'Number of Latrines'
_var493 = df_18[_var492]
_var494 = 1
_var495 = (_var493 > _var494)
_var496 = 'Number of Latrines'
_var497 = df_18[_var496]
_var498 = 2
_var499 = (_var497 <= _var498)
_var500 = (_var495 & _var499)
_var501 = 'LatrinesBandNum'
_var502 = (_var500, _var501)
_var503 = 1
_var491_0 = set_index_wrapper(_var491, _var502, _var503)
_var504 = df_18.loc
_var505 = 'Number of Latrines'
_var506 = df_18[_var505]
_var507 = 2
_var508 = (_var506 > _var507)
_var509 = 'Number of Latrines'
_var510 = df_18[_var509]
_var511 = 5
_var512 = (_var510 <= _var511)
_var513 = (_var508 & _var512)
_var514 = 'LatrinesBandNum'
_var515 = (_var513, _var514)
_var516 = 2
_var504_0 = set_index_wrapper(_var504, _var515, _var516)
_var517 = df_18.loc
_var518 = 'Number of Latrines'
_var519 = df_18[_var518]
_var520 = 5
_var521 = (_var519 > _var520)
_var522 = 'LatrinesBandNum'
_var523 = (_var521, _var522)
_var524 = 3
_var517_0 = set_index_wrapper(_var517, _var523, _var524)
_var525 = 'LatrinesBandNum'
_var526 = 'LatrinesBandNum'
_var527 = df_18[_var526]
_var528 = _var527.astype(int)
df_19 = set_index_wrapper(df_18, _var525, _var528)
_var529 = ['LatrinesBand']
_var530 = 1
df_20 = df_19.drop(_var529, axis=_var530)
df_20.head()
_var531 = 'WaterBand'
_var532 = 'Water Capacity in L'
_var533 = df_20[_var532]
_var534 = 4
_var535 = pd.qcut(_var533, _var534)
df_21 = set_index_wrapper(df_20, _var531, _var535)
_var536 = ['WaterBand', 'Status']
_var537 = df_21[_var536]
_var538 = ['WaterBand']
_var539 = False
_var540 = _var537.groupby(_var538, as_index=_var539)
_var541 = _var540.mean()
_var542 = 'WaterBand'
_var543 = True
_var541.sort_values(by=_var542, ascending=_var543)
_var544 = df_21.loc
_var545 = 'Water Capacity in L'
_var546 = df_21[_var545]
_var547 = 1000
_var548 = (_var546 <= _var547)
_var549 = 'WaterBandNum'
_var550 = (_var548, _var549)
_var551 = 0
_var544_0 = set_index_wrapper(_var544, _var550, _var551)
_var552 = df_21.loc
_var553 = 'Water Capacity in L'
_var554 = df_21[_var553]
_var555 = 1000
_var556 = (_var554 > _var555)
_var557 = 'Water Capacity in L'
_var558 = df_21[_var557]
_var559 = 2500
_var560 = (_var558 <= _var559)
_var561 = (_var556 & _var560)
_var562 = 'WaterBandNum'
_var563 = (_var561, _var562)
_var564 = 1
_var552_0 = set_index_wrapper(_var552, _var563, _var564)
_var565 = df_21.loc
_var566 = 'Water Capacity in L'
_var567 = df_21[_var566]
_var568 = 2500
_var569 = (_var567 > _var568)
_var570 = 'Water Capacity in L'
_var571 = df_21[_var570]
_var572 = 7000
_var573 = (_var571 <= _var572)
_var574 = (_var569 & _var573)
_var575 = 'WaterBandNum'
_var576 = (_var574, _var575)
_var577 = 2
_var565_0 = set_index_wrapper(_var565, _var576, _var577)
_var578 = df_21.loc
_var579 = 'Water Capacity in L'
_var580 = df_21[_var579]
_var581 = 7000
_var582 = (_var580 > _var581)
_var583 = 'WaterBandNum'
_var584 = (_var582, _var583)
_var585 = 3
_var578_0 = set_index_wrapper(_var578, _var584, _var585)
_var586 = 'WaterBandNum'
_var587 = 'WaterBandNum'
_var588 = df_21[_var587]
_var589 = _var588.astype(int)
df_22 = set_index_wrapper(df_21, _var586, _var589)
_var590 = ['WaterBand']
_var591 = 1
df_23 = df_22.drop(_var590, axis=_var591)
df_23.head()
_var592 = ['Status', 'Waste Water Disposal', 'Type of Water Source', 'Type of Contract', 'Type of Internet Connection', 'LatrinesBandNum', 'WaterBandNum', 'Governorate']
df1 = df_23[_var592]
df1.head()
_var593 = df1.iloc
_var594 = 1
df_x = _var593[:, _var594:]
_var595 = df1.iloc
_var596 = 0
df_y = _var595[:, _var596]
_var601 = 0.2
_var602 = 4
(_var597, _var598, _var599, _var600) = train_test_split(df_x, df_y, test_size=_var601, random_state=_var602)
x_train = _var597
x_test = _var598
y_train = _var599
y_test = _var600
decision_tree = DecisionTreeClassifier()
decision_tree_0 = decision_tree.fit(x_train, y_train)
y_pred = decision_tree_0.predict(x_test)
_var603 = decision_tree_0.score(x_train, y_train)
_var604 = 100
_var605 = (_var603 * _var604)
_var606 = 2
acc_decision_tree = round(_var605, _var606)
acc_decision_tree
_var607 = 3
knn = KNeighborsClassifier(n_neighbors=_var607)
knn_0 = knn.fit(x_train, y_train)
y_pred_0 = knn_0.predict(x_test)
_var608 = knn_0.score(x_train, y_train)
_var609 = 100
_var610 = (_var608 * _var609)
_var611 = 2
acc_knn = round(_var610, _var611)
acc_knn
_var612 = 100
random_forest = RandomForestClassifier(n_estimators=_var612)
RF_fit = random_forest.fit(x_train, y_train)
random_forest_0 = RF_fit
y_pred_1 = RF_fit.predict(x_test)
random_forest_0.score(x_train, y_train)
_var613 = random_forest_0.score(x_train, y_train)
_var614 = 100
_var615 = (_var613 * _var614)
_var616 = 2
acc_random_forest = round(_var615, _var616)
acc_random_forest
_var617 = 10
scores = cross_val_score(random_forest_0, x_train, y_train, cv=_var617)
_var618 = 'Accuracy: %0.2f (+/- %0.2f)'
_var619 = scores.mean()
_var620 = scores.std()
_var621 = 2
_var622 = (_var620 * _var621)
_var623 = (_var619, _var622)
_var624 = (_var618 % _var623)
print(_var624)
from sklearn import preprocessing
_var629 = 0.4
_var630 = 0
(_var625, _var626, _var627, _var628) = train_test_split(df_x, df_y, test_size=_var629, random_state=_var630)
X_train = _var625
X_test = _var626
y_train_0 = _var627
y_test_0 = _var628
_var631 = preprocessing.StandardScaler()
scaler = _var631.fit(X_train)
_var631_0 = scaler
X_train_transformed = scaler.transform(X_train)
clf = random_forest_0.fit(X_train_transformed, y_train_0)
random_forest_1 = clf
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test_0)
from sklearn.pipeline import make_pipeline
_var632 = preprocessing.StandardScaler()
_var633 = 1
_var634 = SVC(C=_var633)
clf_0 = make_pipeline(_var632, _var634)
_var635 = 10
cross_val_score(clf_0, df_x, df_y, cv=_var635)
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
_var636 = 10
predicted = cross_val_predict(clf_0, df_x, df_y, cv=_var636)
metrics.accuracy_score(df_y, predicted)
from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
_var637 = 10
_var638 = 10
_var639 = (_var637, _var638)
fig_1 = plt.figure(figsize=_var639)
X = df_x
y_2 = df_y
_var642 = X.shape
_var643 = 0
n_samples = _var642[_var643]
_var644 = 1
n_features = _var642[_var644]
_var645 = np.random
_var646 = 0
random_state = _var645.RandomState(_var646)
_var647 = np.c_
_var648 = 200
_var649 = (_var648 * n_features)
_var650 = random_state.randn(n_samples, _var649)
_var651 = (X, _var650)
X_0 = _var647[_var651]
_var652 = 6
cv = StratifiedKFold(n_splits=_var652)
_var653 = 100
classifier = RandomForestClassifier(n_estimators=_var653, random_state=random_state)
mean_tpr = 0.0
_var654 = 0
_var655 = 1
_var656 = 100
mean_fpr = np.linspace(_var654, _var655, _var656)
_var657 = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
colors = cycle(_var657)
lw = 2
i = 0
_var658 = cv.split(X_0, y_2)
_var659 = zip(_var658, colors)
for _var660 in _var659:
    _var665 = 0
    _var666 = _var660[_var665]
    _var667 = 0
    train = _var666[_var667]
    _var668 = 1
    test = _var666[_var668]
    _var669 = 1
    color = _var660[_var669]
    _var670 = X_0[train]
    _var671 = y_2[train]
    _var672 = classifier.fit(_var670, _var671)
    _var673 = X_0[test]
    probas_ = _var672.predict_proba(_var673)
    _var677 = y_2[test]
    _var678 = 1
    _var679 = probas_[:, _var678]
    (_var674, _var675, _var676) = roc_curve(_var677, _var679)
    fpr = _var674
    tpr = _var675
    thresholds = _var676
    _var680 = interp(mean_fpr, fpr, tpr)
    mean_tpr_0 = (mean_tpr + _var680)
    _var681 = 0
    _var682 = 0.0
    mean_tpr_1 = set_index_wrapper(mean_tpr_0, _var681, _var682)
    roc_auc = auc(fpr, tpr)
    _var683 = 'ROC fold %d (area = %0.2f)'
    _var684 = (i, roc_auc)
    _var685 = (_var683 % _var684)
    plt.plot(fpr, tpr, lw=lw, color=color, label=_var685)
    _var686 = 1
    i_0 = (i + _var686)
i_1 = __phi__(i_0, i)
mean_tpr_2 = __phi__(mean_tpr_1, mean_tpr)
_var687 = [0, 1]
_var688 = [0, 1]
_var689 = '--'
_var690 = 'k'
_var691 = 'Luck'
plt.plot(_var687, _var688, linestyle=_var689, lw=lw, color=_var690, label=_var691)
_var692 = cv.get_n_splits(X_0, y_2)
mean_tpr_3 = (mean_tpr_2 / _var692)
_var693 = (- 1)
_var694 = 1.0
mean_tpr_4 = set_index_wrapper(mean_tpr_3, _var693, _var694)
mean_auc = auc(mean_fpr, mean_tpr_4)
_var695 = 'g'
_var696 = '--'
_var697 = 'Mean ROC (area = %0.2f)'
_var698 = (_var697 % mean_auc)
plt.plot(mean_fpr, mean_tpr_4, color=_var695, linestyle=_var696, label=_var698, lw=lw)
_var699 = (- 0.05)
_var700 = [_var699, 1.05]
plt.xlim(_var700)
_var701 = (- 0.05)
_var702 = [_var701, 1.05]
plt.ylim(_var702)
_var703 = 'False Positive Rate'
plt.xlabel(_var703)
_var704 = 'True Positive Rate'
plt.ylabel(_var704)
_var705 = 'Receiver Operating Characteristic curves'
plt.title(_var705)
_var706 = 'lower right'
plt.legend(loc=_var706)
plt.show()
_var707 = df1.columns
_var708 = 0
_var709 = _var707.delete(_var708)
coeff_df = pd.DataFrame(_var709)
_var710 = ['Feature']
coeff_df_0 = set_field_wrapper(coeff_df, 'columns', _var710)
_var711 = 'Correlation'
_var712 = random_forest_1.feature_importances_
_var713 = pd.Series(_var712)
coeff_df_1 = set_index_wrapper(coeff_df_0, _var711, _var713)
_var714 = 'Correlation'
_var715 = False
_var716 = coeff_df_1.sort_values(by=_var714, ascending=_var715)
print(_var716)
importances = random_forest_1.feature_importances_
_var717 = [tree.feature_importances_ for tree in random_forest.estimators_]
_var718 = 0
std = np.std(_var717, axis=_var718)
_var719 = np.argsort(importances)
_var720 = (- 1)
indices = _var719[::_var720]
plt.figure()
_var721 = 'Feature importances'
plt.title(_var721)
_var722 = x_train.shape
_var723 = 1
_var724 = _var722[_var723]
_var725 = range(_var724)
_var726 = list(_var725)
_var727 = importances[indices]
_var728 = 'r'
_var729 = std[indices]
_var730 = 'center'
plt.bar(_var726, _var727, color=_var728, yerr=_var729, align=_var730)
_var731 = x_train.shape
_var732 = 1
_var733 = _var731[_var732]
_var734 = range(_var733)
_var735 = list(_var734)
_var736 = x_train.columns
_var737 = _var736[indices]
_var738 = _var737.values
_var739 = 'vertical'
plt.xticks(_var735, _var738, rotation=_var739)
_var740 = (- 1)
_var741 = x_train.shape
_var742 = 1
_var743 = _var741[_var742]
_var744 = [_var740, _var743]
plt.xlim(_var744)
_var745 = 'Correlation'
plt.ylabel(_var745)
plt.show()
