# The XGBoost Python module is able to load data from many different types of data format including both CPU and GPU data structures. 
    # The input data is stored in a DMatrix object. 

# load a NumPy array into DMatri x
data = np.random.rand(5, 10) # 5 entities, each contains 10 features. Generates a 5x10 matrix with random values between 0 and 1 (by default)
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix(data, label=label)

# To load a Pandas data frame into DMatrix:
data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c']) # Create a Pandas DataFrame with values from 0 to 11, reshaped into a 4x3 matrix
label = pandas.DataFrame(np.random.randint(2, size=4)) # Create a Pandas DataFrame with 4 random binary labels (0 or 1)
dtrain = xgb.DMatrix(data, label=label)

# To load a CSV file into DMatrix:
    # label_column specifies the index of the column containing the true label
dtrain = xgb.DMatrix('train.csv?format=csv&label_column=0')
dtest = xgb.DMatrix('test.csv?format=csv&label_column=0')

# ----------------------------------------------------------------------------------------------- # 

# Saving the DMatrix in a XGBoost binary file will make loading the DMatrix faster
dtrain = xgb.DMatrix('train.svm.txt?format=libsvm')
dtrain.save_binary('train.buffer')