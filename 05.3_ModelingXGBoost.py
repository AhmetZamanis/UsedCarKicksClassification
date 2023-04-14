# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
from xgboost import XGBClassifier
from torch.cuda import is_available


# Check if GPU is available
is_available()


# Retrieve positive class weight
pos_weight = y_train[y_train == 0].count() / y_train[y_train == 1].count()


# Define XGBoost classifier
model_xgb = XGBClassifier(
  objective = "binary:logistic",
  scale_pos_weight = pos_weight,
  n_estimators = 5000,
  early_stopping_rounds = 50,
  eval_metric = "logloss",
  tree_method = "gpu_hist",
  gpu_id = 0,
  random_state = 1923
)


# Perform train-validation split
x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
  x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
)


# Perform preprocessing
x_train_xgb = pipe_process.fit_transform(x_train_xgb, y_train_xgb)
x_val_xgb = pipe_process.transform(x_val_xgb)


# Test run
model_xgb.fit(X = x_train_xgb, y = y_train_xgb, eval_set = (x_val_xgb, y_val_xgb))
# Fix: ValueError: too many values to unpack (expected 2)


# Define objective function for hyperparameter tuning
def objective_svm(trial):
  
  # Define parameter ranges to tune over


  # Set parameters for trial
  model_xgb.set_params(
    model_xgb__alpha = 
    model_xgb__
    )
  
  # Train model with eval_set
  
  
  # Retrieve best score

  # Return mean score
  return np.mean(scores)









