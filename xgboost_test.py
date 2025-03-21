import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# import XGBoost
import xgboost as xgb
from xgboost import cv

#accuracy_score
from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')

data = './Wholesale customers data.csv'

df = pd.read_csv(data)

df.shape
# (440, 8)
df.head()

df.describe()

df.isnull().sum()

X = df.drop('Channel', axis=1)
y = df['Channel']

X.head()
y.head()

y[y == 2] = 0
y[y == 1] = 1

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# import XGBClassifier
from xgboost import XGBClassifier


# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }
            
            
            
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)



# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

#XGBClassifier(alpha=10, base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=1.0,
#       max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
#       n_estimators=100, n_jobs=1, nthread=None,
#       objective='binary:logistic', random_state=0, reg_alpha=0,
#       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#      subsample=1, verbosity=1)


# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy scorefrom 
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# XGBoost model accuracy score: 0.9167 (91.67%)
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


xgb.plot_importance(xgb_clf, importance_type="weight")
plt.rcParams['figure.figsize'] = [10, 4]
plt.show()