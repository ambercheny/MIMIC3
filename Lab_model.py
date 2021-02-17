import tensorflow as tf 
import os
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy 

# !python -m pip install xgboost
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.keras.backend.set.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#keras.backend.set_session(tf.compat.v1.Session(config=config))

sess = tf.compat.v1.keras.backend.get_session()

task_df = pd.read_csv(r'C:/Users/amber/Desktop/mimic/processed/lab_task.csv')
sub_df= pd.read_csv(r'C:/Users/amber/Desktop/mimic/processed/lab_submission.csv')

# preprocessing of training set
a=task_df.columns.tolist()
print(a[0:10])
del a[0]
print(a[0:10])
task_df=task_df[a]

### split into train and test
msk = numpy.random.rand(len(task_df)) < 0.8
train_df = task_df[msk]
test_df = task_df[~msk]

### transfer dataframes into arrays
def PreprocessData(raw_df):
    ndarray = raw_df.values
    label = ndarray[:,1] #target
    features = ndarray[:,2:] #input
    
    from sklearn import preprocessing
    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    scaledFeatures = minmax_scale.fit_transform(features)
    return scaledFeatures, label

train_features, train_label  = PreprocessData(train_df)
test_features, test_label = PreprocessData(test_df)
print(train_features[:2])
print(train_label[:5])
print(len(train_label))
print(len(test_label))

print('total: ', len(task_df),
     'train: ', len(train_df),
     'test: ', len(test_df))

train_label_1=[]
for i in train_label:
    m = int(i)
    train_label_1.append(m)
train_label_1 = numpy.array(train_label_1)

test_label_1 = []
for i in test_label:
    m = int(i)
    test_label_1.append(m)
test_label_1 = numpy.array(test_label_1)

### preprocessing of evaluation set
b=sub_df.columns.tolist()
print(b[0:10])
del b[0]
print(b[0:10])
sub_df=sub_df[b]

ndarray = sub_df.values
features = ndarray[:,1:] #input
minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
scaledFeatures = minmax_scale.fit_transform(features)

### Model1
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

model = XGBClassifier()

model.fit(train_features,train_label_1)
y_pred = model.predict(test_features)
# scores = cross_val_score(model, task_label, features, cv=5)
print(metrics.classification_report(test_label_1, y_pred))
# print(scores)

sub_df['Label1']= model.predict(scaledFeatures)

### Model2
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


from sklearn import metrics
from sklearn import datasets
model = LogisticRegression()
kf=KFold(n_splits=3, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(task_df.HADM_ID):
    x_train = np.array(task_df.iloc[:,2:])[train_index]
    y_train = np.array(task_df.iloc[:,1])[train_index]
    x_test = np.array(task_df.iloc[:,2:])[test_index]
    y_test = np.array(task_df.iloc[:,1])[test_index]

    
    model.fit(train_features,train_label_1)
    expected.extend(y_test) 
    predicted.extend(model.predict(x_test))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

sub_df['Label2']= model.predict(scaledFeatures)

### Model3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model=Sequential()
model.add(Dense(units=40, input_dim=471, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=30, input_dim=400, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=30, input_dim=100, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
print(model.summary())

%%time
model.compile(loss='binary_crossentropy',
             optimizer='SGD',
             metrics=['accuracy'])

train_history = model.fit(x=train_features,
                         y=train_label_1,
                         validation_split=0.1,
                         epochs=30, batch_size=100, verbose=2)

sub_df['Label3']= model.predict(scaledFeatures)

### Model4
%%time
model = XGBClassifier()
learning_rate=[0.0001,0.001,0.01,0.1,0.2,0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits = 10, shuffle =True, random_state = 7)
grid_search=GridSearchCV(model,param_grid,scoring='precision_micro',n_jobs=-1,cv=kfold)

grid_result = grid_search.fit(train_features,train_label_1)
print('Best: %f using %s' % (grid_result.best_score_,grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['params']
params = grid_ressult.cv_results_['params']
for mean,stdev,param in zip(means,stds,params):
    print("%f (%f) with %r" % (mean,stdev,param))