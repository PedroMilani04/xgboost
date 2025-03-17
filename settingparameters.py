# XGBoost can use either a list of pairs or a dictionary to set parameters. 

# Booster Parameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

# You can also specify multiple eval metrics
param['eval_metric'] = ['auc', 'ams@0']

# alternatively:
# plst = param.items()
# plst += [('eval_metric', 'ams@0')]


# Specify validations set to watch performance
evallist = [(dtrain, 'train'), (dtest, 'eval')]

