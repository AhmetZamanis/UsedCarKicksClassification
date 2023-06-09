# Used cars kicks classification - Modeling, Elastic Net Logistic Regression
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/04_Preprocessing.py").read())


import optuna
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss


# Get train-test indices (3 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define model validation function
def validate_logistic(alpha, l1_ratio, trial, tol = 1e-4, verbose = 0):
  
  # Record the CV scores of the parameter set
  cv_scores = []
  
  # Record best epochs for each CV fold
  best_epochs = []
  
  for i, (train_index, val_index) in enumerate(cv_indices):
  
    # Split training-validation data
    x_tr = x_train.iloc[train_index, ]
    y_tr = y_train.iloc[train_index, ]
    x_val = x_train.iloc[val_index, ]
    y_val = y_train.iloc[val_index, ]
    
    # Perform preprocessing
    x_tr = pipe_process.fit_transform(x_tr, y_tr)
    x_val = pipe_process.transform(x_val)
    
    # Compute class weight
    classes = list(set(y_tr))
    class_weight = compute_class_weight("balanced", classes = classes, y = y_tr)
    sample_weight = np.where(y_tr == 1, class_weight[1], class_weight[0])
    sample_weight_val = np.where(y_val == 1, class_weight[1], class_weight[0])
    
    # Define Logistic Regression classifier with SGD
    model_logistic = SGDClassifier(
      loss = "log_loss", # Log loss for probabilistic logistic regression
      penalty = "elasticnet",
      learning_rate = "optimal", # Dynamically adjusted based on regularization strength 
      random_state = 1923,
      verbose = verbose,
      n_jobs = -1,
      alpha = alpha,
      l1_ratio = l1_ratio
    )
    
    # Perform epoch by epoch training with early stopping & pruning
    epoch_scores = []
    n_iter_no_change = 0
    tol = tol
    
    for epoch in range(1000):
      
      # Train model for 1 epoch
      model_logistic.partial_fit(
        x_tr, y_tr, classes = classes, sample_weight = sample_weight)
      
      # Score epoch
      y_pred = model_logistic.predict_proba(x_val)
      epoch_score = log_loss(y_val, y_pred, sample_weight = sample_weight_val)
      
      # For first CV fold, report intermediate score of trial to Optuna
      if i == 0:
        trial.report(epoch_score, epoch)
      
        # Prune trial if necessary
        if trial.should_prune():
          raise optuna.TrialPruned()
      
      # Count epochs with no improvement after first 10 epochs
      if (epoch > 9) and (epoch_score > min(epoch_scores) - tol):
        n_iter_no_change += 1
      
      # Reset epochs with no improvement if an improvement took place
      if (epoch > 9) and (epoch_score <= min(epoch_scores) - tol):
        n_iter_no_change = 0
      
      # Append epoch score to list of epoch scores
      epoch_scores.append(epoch_score)
      
      # Early stop training if necessary
      if n_iter_no_change >= 10:
        print("\nEarly stopping at epoch " + str(epoch) + "\n")
        break 
     
    # Append best epoch score to list of CV scores
    cv_scores.append(min(epoch_scores))
    
    # Append best epoch number to list of best epochs
    best_epochs.append(epoch_scores.index(min(epoch_scores)) + 1)
  
  # Return the average CV score & median best epochs
  return np.mean(cv_scores), np.median(best_epochs)
  

# Define objective function for hyperparameter tuning
def objective_logistic(trial):
  
  # Define parameter ranges to tune over
  alpha = trial.suggest_float("reg_strength", 5e-5, 0.5, log = True)
  l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
  
  # Validate the parameter set
  score, epoch = validate_logistic(
    alpha = alpha, l1_ratio = l1_ratio, trial = trial)
    
  # Report best n. of epochs to Optuna
  trial.set_user_attr("n_epochs", epoch)
  
  return score


# Create study
study_logistic = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_logistic",
  direction = "minimize"
)


# Optimize study
study_logistic.optimize(
  objective_logistic, 
  n_trials = 500,
  show_progress_bar = True)


# Retrieve and export trials
trials_logistic = study_logistic.trials_dataframe().sort_values("value", ascending = True)
trials_logistic.to_csv("./ModifiedData/trials_logistic.csv", index = False)

