# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/04_Preprocessing.py").read())


import optuna
from xgboost import XGBClassifier
from torch.cuda import is_available
from sklearn.utils.class_weight import compute_class_weight


# Check if GPU is available
is_available()


# Get train-test indices (3 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define model validation function
def validate_xgb(params_dict, trial, verbose = 0):
  
  # Record best epoch scores for each CV fold
  cv_scores = []
  
  # Record best epochs for each CV fold
  best_epochs = []
  
  for i, (train_index, val_index) in enumerate(cv_indices):
  
    # Split training-validation data
    x_tr = x_train.iloc[train_index, ]
    y_tr = y_train.iloc[train_index, ]
    x_val = x_train.iloc[val_index, ]
    y_val = y_train.iloc[val_index, ]
    
    # Compute class weight
    classes = list(set(y_tr))
    class_weight = compute_class_weight("balanced", classes = classes, y = y_tr)
    sample_weight = np.where(y_tr == 1, class_weight[1], class_weight[0])
    sample_weight_val = np.where(y_val == 1, class_weight[1], class_weight[0])
    
    # Perform preprocessing
    x_tr = pipe_process.fit_transform(x_tr, y_tr)
    x_val = pipe_process.transform(x_val)
    
    # Create pruning callback for first CV split
    if i == 0: 
      callback_pruner = [optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-logloss")]
    
    else:
      callback_pruner = None
    
    # Define XGBoost classifier
    model_xgb = XGBClassifier(
        objective = "binary:logistic",
        n_estimators = 5000,
        early_stopping_rounds = 50,
        eval_metric = "logloss",
        tree_method = "gpu_hist",
        gpu_id = 0,
        verbosity = 0,
        random_state = 1923,
        callbacks = callback_pruner,
        learning_rate = params_dict["learning_rate"],
        max_depth = params_dict["max_depth"],
        min_child_weight = params_dict["min_child_weight"],
        gamma = params_dict["gamma"],
        reg_alpha = params_dict["reg_alpha"],
        reg_lambda = params_dict["reg_lambda"],
        subsample = params_dict["subsample"],
        colsample_bytree = params_dict["colsample_bytree"]
        )
        
    # Train & validate model
    model_xgb.fit(
      X = x_tr, 
      y = y_tr, 
      sample_weight = sample_weight,
      eval_set = [(x_val, y_val)], 
      sample_weight_eval_set = [sample_weight_val],
      verbose = verbose)
    
    # Append best epoch score to list of CV scores
    cv_scores.append(model_xgb.best_score)
    
    # Append best epoch number to list of best epochs
    best_epochs.append(model_xgb.best_iteration + 1)
  
  # Return the average CV score & median best n. rounds
  return np.mean(cv_scores), np.median(best_epochs)
  
 
# Define objective function for hyperparameter tuning
def objective_xgb(trial):
  
  # Suggest parameter values from parameter ranges to tune over
  learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3)
  max_depth = trial.suggest_int("max_depth", 2, 20)
  min_child_weight = trial.suggest_int("min_child_weight", 1, 20, log = True)
  gamma = trial.suggest_float("gamma", 5e-5, 0.5, log = True)
  reg_alpha = trial.suggest_float("l1_reg", 5e-5, 1, log = True)
  reg_lambda = trial.suggest_float("l2_reg", 0, 2)
  subsample = trial.suggest_float("subsample", 0.5, 1)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.25, 1)
  
  # Make dictionary of parameters
  params_dict = {
    "learning_rate": learning_rate,
    "max_depth": max_depth,
    "min_child_weight": min_child_weight,
    "gamma": gamma,
    "reg_alpha": reg_alpha,
    "reg_lambda": reg_lambda,
    "subsample": subsample,
    "colsample_bytree": colsample_bytree
  }
  
  # Validate parameter set
  score, rounds = validate_xgb(
    params_dict = params_dict, trial = trial)
    
  # Report best n. rounds to Optuna
  trial.set_user_attr("n_rounds", rounds)
  
  return score


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
  show_progress_bar = True)


# Retrieve and export trials
trials_xgb = study_xgb.trials_dataframe().sort_values("value", ascending = True)
trials_xgb.to_csv("./ModifiedData/trials_xgb.csv", index = False)
