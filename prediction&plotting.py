#Ealy Stopping / Prediction

#If you have a validation set, you can use early stopping to find the optimal number of boosting rounds. 
    #Early stopping requires at least one set in evals. If there’s more than one, it will use the last.
train(..., evals=evals, early_stopping_rounds=10)

#If early stopping occurs, the model will have two additional fields: bst.best_score, bst.best_iteration. 
    #Note that xgboost.train() will return a model from the last iteration, not the best one.


# A model that has been trained or loaded can perform predictions on data sets.
    # 7 entities, each contains 10 features
data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)

# If early stopping is enabled during training, you can get predictions from the best iteration with bst.best_iteration:
ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

# Plotting

# To plot importance, use xgboost.plot_importance(). This function requires matplotlib to be installed.
xgb.plot_importance(bst)

# To plot the output tree via matplotlib, use xgboost.plot_tree(), specifying the ordinal number of the target tree. This function requires graphviz and matplotlib.
xgb.plot_tree(bst, num_trees=2)




