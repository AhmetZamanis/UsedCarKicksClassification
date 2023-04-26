# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
from xgboost import XGBClassifier
from torch.cuda import is_available
from sklearn.utils.class_weight import compute_class_weight


# Check if GPU is available
is_available()


# Get train-test indices (3 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define objective function for hyperparameter tuning
def objective_xgb(trial):
  
  # Suggest parameter values from parameter ranges to tune over
  learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3)
  max_depth = trial.suggest_int("max_depth", 2, 20)
  min_child_weight = trial.suggest_int("min_child_weight", 1, 20, log = True)
  gamma = trial.suggest_float("gamma", 0.01, 0.5, log = True)
  reg_alpha = trial.suggest_float("l1_reg", 0.01, 1, log = True)
  reg_lambda = trial.suggest_float("l2_reg", 0.05, 2)
  subsample = trial.suggest_float("subsample", 0.5, 1)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
  
  # Crossvalidate the parameter set
  cv_scores = []
  
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
    
    if i == 0: 
      
      # Create pruning callback for first CV split
      callback_pruner = optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-logloss")
    
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
    model_xgb.fit(
      X = x_tr, 
      y = y_tr, 
      sample_weight = sample_weight,
      eval_set = [(x_val, y_val)], 
      sample_weight_eval_set = [sample_weight_val],
      verbose = False)
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
  n_trials = 1000,
  n_jobs = 1,
  show_progress_bar = True)


# Retrieve and export trials
trials_xgb = study_xgb.trials_dataframe().sort_values("value", ascending = True)
trials_xgb.to_csv("./ModifiedData/trials_xgb2.csv", index = False)


# Import best trial
best_trial_xgb = pd.read_csv("./ModifiedData/trials_xgb2.csv")
best_trial_xgb = best_trial_xgb.loc[
  best_trial_xgb["state"] == "COMPLETE"].iloc[0,]


# Retrieve best early stop rounds with optimal parameters for each CV fold
best_iterations = []
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
    
    # Define XGB classifier
    model_xgb = XGBClassifier(
        objective = "binary:logistic",
        n_estimators = 5000,
        early_stopping_rounds = 50,
        eval_metric = "logloss",
        tree_method = "gpu_hist",
        gpu_id = 0,
        verbosity = 0,
        random_state = 1923,
        learning_rate = best_trial_xgb["params_learning_rate"],
        max_depth = best_trial_xgb["params_max_depth"],
        min_child_weight = best_trial_xgb["params_min_child_weight"],
        gamma = best_trial_xgb["params_gamma"],
        reg_alpha = best_trial_xgb["params_l1_reg"],
        reg_lambda = best_trial_xgb["params_l2_reg"],
        subsample = best_trial_xgb["params_subsample"],
        colsample_bytree = best_trial_xgb["params_colsample_bytree"]
        )
        
    # Perform preprocessing
    x_tr = pipe_process.fit_transform(x_tr, y_tr)
    x_val = pipe_process.transform(x_val)

    # Fit model and save best round index
    model_xgb.fit(
      X = x_tr, 
      y = y_tr, 
      sample_weight = sample_weight,
      eval_set = [(x_val, y_val)], 
      sample_weight_eval_set = [sample_weight_val],
      verbose = True)
    best_iterations.append(model_xgb.best_iteration + 1)


# Retrieve median of best n_estimators: 18 rounds
int(np.median(best_iterations))
