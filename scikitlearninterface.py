# XGBoost provides an easy to use scikit-learn interface for some pre-defined models including regression, classification and ranking. 
    # Use "hist" for training the model.
        # It's XGBoost, but using functions like fit and predict, like scikit models.
reg = xgb.XGBRegressor(tree_method="hist", device="cuda") #histogram-based optimization / runs the training in the GPU if available
    # Fit the model using predictor X and response y.
reg.fit(X, y)
    # Save model into JSON format.
reg.save_model("regressor.json")

# User can still access the underlying booster model when needed:
booster: xgb.Booster = reg.get_booster()
