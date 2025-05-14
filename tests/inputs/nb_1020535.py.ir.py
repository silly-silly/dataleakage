

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
'\nSolution\n'
import pandas as pd
_var0 = 'smsspamcollection/SMSSpamCollection'
_var1 = '\t'
_var2 = None
_var3 = ['label', 'sms_message']
df = pd.read_table(_var0, sep=_var1, header=_var2, names=_var3)
df.head()
'\nSolution\n'
_var4 = 'label'
_var5 = df.label
_var6 = {'ham': 0, 'spam': 1}
_var7 = _var5.map(_var6)
df_0 = set_index_wrapper(df, _var4, _var7)
_var8 = df_0.shape
print(_var8)
df_0.head()
'\nSolution:\n'
documents = ['Hello, how are you!', 'Win money, win from home.', 'Call me now.', 'Hello, Call hello you tomorrow?']
lower_case_documents = []
for i in documents:
    _var9 = i.lower()
    lower_case_documents.append(_var9)
print(lower_case_documents)
'\nSolution:\n'
sans_punctuation_documents = []
import string
for i_0 in lower_case_documents:
    _var10 = ''
    _var11 = ''
    _var12 = string.punctuation
    _var13 = str.maketrans(_var10, _var11, _var12)
    _var14 = i_0.translate(_var13)
    sans_punctuation_documents.append(_var14)
print(sans_punctuation_documents)
'\nSolution:\n'
preprocessed_documents = []
for i_1 in sans_punctuation_documents:
    _var15 = ' '
    _var16 = i_1.split(_var15)
    preprocessed_documents.append(_var16)
print(preprocessed_documents)
'\nSolution\n'
frequency_list = []
import pprint
from collections import Counter
for i_2 in preprocessed_documents:
    frequency_counts = Counter(i_2)
    frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)
"\nHere we will look to create a frequency matrix on a smaller document set to make sure we understand how the \ndocument-term matrix generation happens. We have created a sample document set 'documents'.\n"
documents_0 = ['Hello, how are you!', 'Win money, win from home.', 'Call me now.', 'Hello, Call hello you tomorrow?']
'\nSolution\n'
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
"\nPractice node:\nPrint the 'count_vector' object which is an instance of 'CountVectorizer()'\n"
print(count_vector)
'\nSolution:\n'
count_vector_0 = count_vector.fit(documents_0)
count_vector_0.get_feature_names()
'\nSolution\n'
_var17 = count_vector_0.transform(documents_0)
doc_array = _var17.toarray()
doc_array
'\nSolution\n'
_var18 = count_vector_0.get_feature_names()
frequency_matrix = pd.DataFrame(doc_array, columns=_var18)
frequency_matrix
'\nSolution\n\nNOTE: sklearn.cross_validation will be deprecated soon to sklearn.model_selection \n'
from sklearn.cross_validation import train_test_split
_var23 = 'sms_message'
_var24 = df_0[_var23]
_var25 = 'label'
_var26 = df_0[_var25]
_var27 = 1
(_var19, _var20, _var21, _var22) = train_test_split(_var24, _var26, random_state=_var27)
X_train = _var19
X_test = _var20
y_train = _var21
y_test = _var22
_var28 = 'Number of rows in the total set: {}'
_var29 = df_0.shape
_var30 = 0
_var31 = _var29[_var30]
_var32 = _var28.format(_var31)
print(_var32)
_var33 = 'Number of rows in the training set: {}'
_var34 = X_train.shape
_var35 = 0
_var36 = _var34[_var35]
_var37 = _var33.format(_var36)
print(_var37)
_var38 = 'Number of rows in the test set: {}'
_var39 = X_test.shape
_var40 = 0
_var41 = _var39[_var40]
_var42 = _var38.format(_var41)
print(_var42)
"\n[Practice Node]\n\nThe code for this segment is in 2 parts. Firstly, we are learning a vocabulary dictionary for the training data \nand then transforming the data into a document-term matrix; secondly, for the testing data we are only \ntransforming the data into a document-term matrix.\n\nThis is similar to the process we followed in Step 2.3\n\nWe will provide the transformed data to students in the variables 'training_data' and 'testing_data'.\n"
'\nSolution\n'
count_vector_1 = CountVectorizer()
training_data = count_vector_1.fit_transform(X_train)
testing_data = count_vector_1.transform(X_test)
'\nInstructions:\nCalculate probability of getting a positive test result, P(Pos)\n'
'\nSolution (skeleton code will be provided)\n'
p_diabetes = 0.01
p_no_diabetes = 0.99
p_pos_diabetes = 0.9
p_neg_no_diabetes = 0.9
_var43 = (p_diabetes * p_pos_diabetes)
_var44 = 1
_var45 = (_var44 - p_neg_no_diabetes)
_var46 = (p_no_diabetes * _var45)
p_pos = (_var43 + _var46)
_var47 = 'The probability of getting a positive test result P(Pos) is: {}'
_var48 = format(p_pos)
_var49 = (_var47, _var48)
print(_var49)
'\nInstructions:\nCompute the probability of an individual having diabetes, given that, that individual got a positive test result.\nIn other words, compute P(D|Pos).\n\nThe formula is: P(D|Pos) = (P(D) * P(Pos|D) / P(Pos)\n'
'\nSolution\n'
_var50 = (p_diabetes * p_pos_diabetes)
p_diabetes_pos = (_var50 / p_pos)
_var51 = 'Probability of an individual having diabetes, given that that individual got a positive test result is:'
_var52 = format(p_diabetes_pos)
_var53 = (_var51, _var52)
print(_var53)
'\nInstructions:\nCompute the probability of an individual not having diabetes, given that, that individual got a positive test result.\nIn other words, compute P(~D|Pos).\n\nThe formula is: P(~D|Pos) = (P(~D) * P(Pos|~D) / P(Pos)\n\nNote that P(Pos/~D) can be computed as 1 - P(Neg/~D). \n\nTherefore:\nP(Pos/~D) = p_pos_no_diabetes = 1 - 0.9 = 0.1\n'
'\nSolution\n'
p_pos_no_diabetes = 0.1
_var54 = (p_no_diabetes * p_pos_no_diabetes)
p_no_diabetes_pos = (_var54 / p_pos)
_var55 = 'Probability of an individual not having diabetes, given that that individual got a positive test result is:'
_var56 = (_var55, p_no_diabetes_pos)
print(_var56)
"\nInstructions: Compute the probability of the words 'freedom' and 'immigration' being said in a speech, or\nP(F,I).\n\nThe first step is multiplying the probabilities of Jill Stein giving a speech with her individual \nprobabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_j_text\n\nThe second step is multiplying the probabilities of Gary Johnson giving a speech with his individual \nprobabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_g_text\n\nThe third step is to add both of these probabilities and you will get P(F,I).\n"
'\nSolution: Step 1\n'
p_j = 0.5
p_j_f = 0.1
p_j_i = 0.1
_var57 = (p_j * p_j_f)
p_j_text = (_var57 * p_j_i)
print(p_j_text)
'\nSolution: Step 2\n'
p_g = 0.5
p_g_f = 0.7
p_g_i = 0.2
_var58 = (p_g * p_g_f)
p_g_text = (_var58 * p_g_i)
print(p_g_text)
'\nSolution: Step 3: Compute P(F,I) and store in p_f_i\n'
p_f_i = (p_j_text + p_g_text)
_var59 = 'Probability of words freedom and immigration being said are: '
_var60 = format(p_f_i)
_var61 = (_var59, _var60)
print(_var61)
'\nInstructions:\nCompute P(J|F,I) using the formula P(J|F,I) = (P(J) * P(F|J) * P(I|J)) / P(F,I) and store it in a variable p_j_fi\n'
'\nSolution\n'
p_j_fi = (p_j_text / p_f_i)
_var62 = 'The probability of Jill Stein saying the words Freedom and Immigration: '
_var63 = format(p_j_fi)
_var64 = (_var62, _var63)
print(_var64)
'\nInstructions:\nCompute P(G|F,I) using the formula P(G|F,I) = (P(G) * P(F|G) * P(I|G)) / P(F,I) and store it in a variable p_g_fi\n'
'\nSolution\n'
p_g_fi = (p_g_text / p_f_i)
_var65 = 'The probability of Gary Johnson saying the words Freedom and Immigration: '
_var66 = format(p_g_fi)
_var67 = (_var65, _var66)
print(_var67)
"\nInstructions:\n\nWe have loaded the training data into the variable 'training_data' and the testing data into the \nvariable 'testing_data'.\n\nImport the MultinomialNB classifier and fit the training data into the classifier using fit(). Name your classifier\n'naive_bayes'. You will be training the classifier using 'training_data' and y_train' from our split earlier. \n"
'\nSolution\n'
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes_0 = naive_bayes.fit(training_data, y_train)
"\nInstructions:\nNow that our algorithm has been trained using the training data set we can now make some predictions on the test data\nstored in 'testing_data' using predict(). Save your predictions into the 'predictions' variable.\n"
'\nSolution\n'
predictions = naive_bayes_0.predict(testing_data)
"\nInstructions:\nCompute the accuracy, precision, recall and F1 scores of your model using your test data 'y_test' and the predictions\nyou made earlier stored in the 'predictions' variable.\n"
'\nSolution\n'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
_var68 = 'Accuracy score: '
_var69 = accuracy_score(y_test, predictions)
_var70 = format(_var69)
_var71 = (_var68, _var70)
print(_var71)
_var72 = 'Precision score: '
_var73 = precision_score(y_test, predictions)
_var74 = format(_var73)
_var75 = (_var72, _var74)
print(_var75)
_var76 = 'Recall score: '
_var77 = recall_score(y_test, predictions)
_var78 = format(_var77)
_var79 = (_var76, _var78)
print(_var79)
_var80 = 'F1 score: '
_var81 = f1_score(y_test, predictions)
_var82 = format(_var81)
_var83 = (_var80, _var82)
print(_var83)
