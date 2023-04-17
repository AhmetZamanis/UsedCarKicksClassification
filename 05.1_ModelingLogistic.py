# Used cars kicks classification - Modeling, Elastic Net Logistic Regression
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


from sklearn.linear_model import LogisticRegression
import optuna


# Define logistic regression
model_logistic = LogisticRegression(
  penalty = "elasticnet",
  class_weight = "balanced",
  solver = "saga",
  max_iter = 1000,
  random_state = 1923
)


# Define logistic regression pipeline
pipe_logistic = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_logistic", model_logistic)
])


# Define objective function for hyperparameter tuning
def objective_logistic(trial):
  
  # Define parameter ranges to tune over
  C = 1 / trial.suggest_float("C_inv", 0.005, 0.5, step = 0.005)
  l1_ratio = trial.suggest_float("l1_ratio", 0, 1, step = 0.05)

  # Set parameters for trial
  pipe_logistic.set_params(
    model_logistic__C = C,
    model_logistic__l1_ratio = l1_ratio
  )
  
  # Perform trial and retrieve cross validation scores
  scores = -1 * cross_val_score(
    estimator = pipe_logistic,
    X = x_train,
    y = y_train,
    scoring = "neg_log_loss",
    cv = cv_kfold
  )

  # Return mean score
  return np.mean(scores)


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
  n_jobs = -1,
  show_progress_bar = True)


# Retrieve and export trials
trials_logistic = study_logistic.trials_dataframe().sort_values("value", ascending = True)
trials_logistic.to_csv("./ModifiedData/trials_logistic.csv", index = False)



