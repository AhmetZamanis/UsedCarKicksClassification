# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss
from sklearn.utils.class_weight import compute_class_weight


# Recode target for hinge loss calculation
y_train.loc[y_train == 0] = -1


# Get train-test indices (10 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define objective function for hyperparameter tuning
def objective_svm(trial):
  
  # Define parameter ranges to tune over
  alpha = trial.suggest_float("reg_strength", 0.0001, 0.5, log = True)
  l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
  
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
    sample_weight = np.where(y_tr == 1, class_weight[0], class_weight[1])
    sample_weight_val = np.where(y_val == 1, class_weight[0], class_weight[1])
    
    # Define RBF kernel approximator
    kernel_rbf = RBFSampler(
      gamma = "scale",
      n_components = 100,
      random_state = 1923
    )
    
    # Define preprocessing & kernel trick pipeline
    pipe_svm = Pipeline(steps = [
      ("preprocessing", pipe_process),
      ("kernel_rbf", kernel_rbf)
    ])
    
    # Define SVM classifier with SGD
    model_svm = SGDClassifier(
      loss = "hinge", # Hinge loss for SVM
      penalty = "elasticnet", # Elastic net penalty as opposed to default L2 for SVM
      learning_rate = "optimal", # Dynamically adjusted based on regularization strength 
      random_state = 1923,
      verbose = 0, # Change to 1 to print epochs for debugging if needed
      alpha = alpha,
      l1_ratio = l1_ratio
    )
    
    # Perform preprocessing
    x_tr = pipe_svm.fit_transform(x_tr, y_tr)
    x_val = pipe_svm.transform(x_val)
    
    # Perform epoch by epoch training with early stopping & pruning
    epoch_scores = []
    n_iter_no_change = 0
    tol = 0.001
    
    for epoch in range(1000):
      
      # Train model for 1 epoch
      model_svm.partial_fit(x_tr, y_tr, classes = classes, sample_weight = sample_weight)
      
      # Score epoch
      pred_decision = model_svm.decision_function(x_val)
      epoch_score = hinge_loss(y_val, pred_decision, sample_weight = sample_weight_val)
      
      # For first CV fold, report intermediate score of trial
      if i == 0:
        trial.report(epoch_score, epoch)
      
        # Prune trial if necessary
        if trial.should_prune():
          raise optuna.TrialPruned()
      
      # Count epochs with no improvement after first 10 epochs
      if epoch > 10:
        if (epoch_score > min(epoch_scores) - tol):
          n_iter_no_change += 1
      
      # Append epoch score to list of epoch scores
      epoch_scores.append(epoch_score)
      
      # Early stop training if necessary
      if n_iter_no_change == 10:
        print("Early stopping at epoch " + str(epoch))
        break 
     
    # Append best epoch score to CV scores
    cv_scores.append(min(epoch_scores))
  
  return np.mean(cv_scores)


# Create study
study_svm = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_svm",
  direction = "minimize"
)


# Optimize study
study_svm.optimize(
  objective_svm, 
  n_trials = 500, 
  show_progress_bar = True)


# Retrieve and export trials
# Huge improvement with trial 476
trials_svm = study_svm.trials_dataframe().sort_values("value", ascending = True)
trials_svm.to_csv("./ModifiedData/trials_svm.csv", index = False)