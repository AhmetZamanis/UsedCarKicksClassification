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


# Get train-test indices (10 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define objective function for hyperparameter tuning
def objective_xgb(trial):
  
  # Suggest parameter values from parameter ranges to tune over
  learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3, step = 0.05)
  max_depth = trial.suggest_int("max_depth", 2, 12, step = 2)
  min_child_weight = trial.suggest_int("min_child_weight", 1, 20, log = True)
  gamma = trial.suggest_float("gamma", 0.01, 0.1, log = True)
  reg_alpha = trial.suggest_float("l1_reg", 0.05, 1, log = True)
  reg_lambda = trial.suggest_float("l2_reg", 0.05, 2, log = True)
  subsample = trial.suggest_float("subsample", 0.8, 1, step = 0.1)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.8, 1, step = 0.1)
  
  # Crossvalidate the parameter set
  cv_scores = []
  
  for i, (train_index, val_index) in enumerate(cv_indices):
  
    # Split training-validation data
    x_tr = x_train.iloc[train_index, ]
    y_tr = y_train.iloc[train_index, ]
    x_val = x_train.iloc[val_index, ]
    y_val = y_train.iloc[val_index, ]
    
    # Retrieve positive class weight
    pos_weight = y_tr[y_tr == 0].count() / y_tr[y_tr == 1].count()
    
    if i == 0: 
      
      # Create pruning callback for first CV split
      callback_pruner = optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-logloss")
    
      # Define XGBoost classifier
      model_xgb = XGBClassifier(
        objective = "binary:logistic",
        scale_pos_weight = pos_weight,
        n_estimators = 5000,
        early_stopping_rounds = 50,
        eval_metric = "logloss",
        tree_method = "gpu_hist",
        gpu_id = 0,
        verbosity = 0,
        random_state = 1923,
        callbacks = [callback_pruner],
        learning_rate = learning_rate,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        subsample = subsample,
        colsample_bytree = colsample_bytree
        )
        
    else:
      
      # Define XGBoost classifier without pruning callback for remaining splits
      model_xgb = XGBClassifier(
        objective = "binary:logistic",
        scale_pos_weight = pos_weight,
        n_estimators = 5000,
        early_stopping_rounds = 50,
        eval_metric = "logloss",
        tree_method = "gpu_hist",
        gpu_id = 0,
        verbosity = 0,
        random_state = 1923,
        learning_rate = learning_rate,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        subsample = subsample,
        colsample_bytree = colsample_bytree
        )
  
    # Perform preprocessing
    x_tr = pipe_process.fit_transform(x_tr, y_tr)
    x_val = pipe_process.transform(x_val)

    # Fit model and save best score achieved with early stopping
    model_xgb.fit(X = x_tr, y = y_tr, eval_set = [(x_val, y_val)], verbose = False)
    cv_scores.append(model_xgb.best_score)

  return np.mean(cv_scores)


# Create study
study_xgb = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_xgb",
  direction = "minimize"
)


# Optimize study
study_xgb.optimize(
  objective_xgb, 
  n_trials = 500,
  n_jobs = 1,
  show_progress_bar = True)


# Retrieve and export trials
trials_xgb = study_xgb.trials_dataframe().sort_values("value", ascending = True)
trials_xgb.to_csv("./ModifiedData/trials_xgb.csv", index = False)

