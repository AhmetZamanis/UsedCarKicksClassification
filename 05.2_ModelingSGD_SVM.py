# Used cars kicks classification - Modeling, Support Vector Machine with SGD
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss, make_scorer


# Recode target for hinge loss calculation
y_train.loc[y_train == 0] = -1


# Make hinge loss scorer
scorer_svm = make_scorer(
  hinge_loss,
  greater_is_better = True 
  # False flips the sign, but hinge loss is already positive. Parameter name misleading.
  )


# Define RBF kernel approximator
kernel_rbf = RBFSampler(
  gamma = "scale",
  n_components = 100,
  random_state = 1923
)


# Define SVM classifier with SGD
model_svm = SGDClassifier(
  loss = "hinge", # Hinge loss for SVM
  penalty = "elasticnet", # Elastic net penalty as opposed to default L2 for SVM
  max_iter = 1000, 
  learning_rate = "optimal", # Dynamically adjusted based on regularization strength 
  early_stopping = True,
  validation_fraction = 0.2,
  tol = 0.001, # Early stopping tolerance threshold for loss
  n_iter_no_change = 10, # Early stopping tolerance iterations
  class_weight = "balanced",
  random_state = 1923,
  verbose = 0 # Change to 1 to print epochs for debugging if needed
)


# Define SVM pipeline
pipe_svm = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("kernel_rbf", kernel_rbf),
  ("model_svm", model_svm)
])


# # Test run: Takes few seconds, trains 10-15 epochs. Early stop seems to be working
# # correctly. Inner val scores go down to 0.68-69 in every fold. Outer val scores 
# # range from 0.72 to 0.4.
#  scores = cross_val_score(
#     estimator = pipe_svm,
#     X = x_train,
#     y = y_train,
#     scoring = scorer_svm,
#     cv = cv_kfold
#   )


# Define objective function for hyperparameter tuning
def objective_svm(trial):
  
  # Define parameter ranges to tune over
  alpha = trial.suggest_float("reg_strength", 0.0001, 0.5, log = True)
  l1_ratio = trial.suggest_float("l1_ratio", 0, 1)

  # Set parameters for trial
  pipe_svm.set_params(
    model_svm__alpha = alpha,
    model_svm__l1_ratio = l1_ratio
    )
  
  # Perform trial and retrieve cross validation scores
  scores = cross_val_score(
     estimator = pipe_svm,
     X = x_train,
     y = y_train,
     scoring = scorer_svm,
     cv = cv_kfold
  )

  # Return mean score
  return np.mean(scores)


# Create study
study_svm = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  study_name = "tune_svm",
  direction = "minimize"
)


# Optimize study
study_svm.optimize(
  objective_svm, 
  n_trials = 500, 
  n_jobs = -1,
  show_progress_bar = True)


# Retrieve and export trials
# Huge improvement with trial 476
trials_svm = study_svm.trials_dataframe().sort_values("value", ascending = True)
trials_svm.to_csv("./ModifiedData/trials_svm.csv", index = False)
