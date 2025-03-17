# Training a model requires a parameter list and data set.
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

bst.save_model('0001.model')

# The model and its feature map can also be dumped to a text file.
    # dump model
bst.dump_model('dump.raw.txt')
    # dump model with feature map
bst.dump_model('dump.raw.txt', 'featmap.txt')

# A saved model can be loaded as follows
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('model.bin')  # load model data
