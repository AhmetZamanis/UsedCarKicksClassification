# Used cars kicks classification - Model scoring
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from xgboost import XGBClassifier


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight_train = np.where(y_train == 1, class_weight[1], class_weight[0])
sample_weight_test = np.where(y_test == 1, class_weight[1], class_weight[0])


# Create dummy classifier pipeline
pipe_dummy = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_dummy", DummyClassifier(strategy = "prior"))
])


# Create logistic regression pipeline with optimal hyperparameters
best_trial_logistic = pd.read_csv("./ModifiedData/trials_logistic.csv").iloc[0,]
pipe_logistic = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_logistic", LogisticRegression(
      penalty = "elasticnet",
      C = (1 / best_trial_logistic["params_reg_strength"]),
      l1_ratio = best_trial_logistic["params_l1_ratio"],
      solver = "saga",
      random_state = 1923,
      max_iter = 1000,
      class_weight = "balanced",
    )
  )
])


# Create SVM pipeline with optimal hyperparameters
best_trial_svm = pd.read_csv("./ModifiedData/trials_svm.csv").iloc[0,]
pipe_svm = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("kernel_rbf", RBFSampler(
      gamma = "scale",
      n_components = 100,
      random_state = 1923
    )
  ),
  ("model_svm", SGDClassifier(
      loss = "hinge",
      penalty = "elasticnet",
      alpha = best_trial_svm["params_reg_strength"],
      l1_ratio = best_trial_svm["params_l1_ratio"],
      max_iter = 1000,
      verbose = 1,
      random_state = 1923,
      early_stopping = True,
      validation_fraction = 0.1,
      n_iter_no_change = 10,
      class_weight = "balanced"
    )
  )
])


# Create XGBoost pipeline with optimal hyperparameters
best_trial_xgb = pd.read_csv("./ModifiedData/trials_xgb.csv").iloc[0,]
pipe_xgb = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_xgb", XGBClassifier(
      objective = "binary:logistic",
      n_estimators = 29,
      eval_metric = "logloss",
      tree_method = "gpu_hist",
      gpu_id = 0,
      verbosity = 1,
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
  )
])


# Make named dictionary of models
models_dict = {
  "Dummy classifier": pipe_dummy,
  "Elastic net logistic": pipe_logistic,
  "SVM with SGD, RBF kernel": pipe_svm,
  "XGBoost": pipe_xgb
}


# Define function that fits every model pipeline & scores test data, plots PRC
# curve
def score_models(models_dict):
  
  # Make dataframe to store precision-recall values for each threshold
  df_prc = pd.DataFrame(
    columns = ["Model", "Precision", "Recall"])
  
  # Make dataframe to store PRAUC & Brier scores
  df_scores = pd.DataFrame(
    columns = ["Model", "PRAUC", "Brier score"]
  )
  
  for key in models_dict.keys():
    
    # Fit model, predict classes & probs for test data
    model = models_dict[key]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    y_prob_pos = np.array([x[1] for x in y_prob])
    
    # Compute PRC values
    precision, recall, threshold = precision_recall_curve(
      y_test, y_prob_pos, pos_label = 1)
    
    # Compute PRAUC
    prauc = auc(recall, precision)
    
    # Compute Brier score
    brier_score = brier_score_loss(
      y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)
    
    # Make dataframe of precision-recall values for each threshold
    prc = pd.DataFrame({
      "Model": np.repeat(key, len(precision)),
      "Precision": precision,
      "Recall": recall
    }, index = [0])
    
    # Concatenate PRC values to full dataframe
    df_prc = pd.concat([df_prc, prc])
    
    # Make dataframe of PRAUC & Brier scores, concatenate to full dataframe
    scores = pd.DataFrame({
      "Model": key,
      "PRAUC": prauc,
      "Brier score": brier_score
    })
    df_scores = pd.concat([df_scores, scores])
  
  return df_prc, df_scores


# Retrieve PRC values & performance scores
df_prc, df_scores = score_models(models_dict)



# Fit on training data, predict testing data, retrieve positive label probs
pipe_logistic.fit(x_train, y_train)
y_pred = pipe_logistic.predict(x_test)
y_prob = pipe_logistic.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])

# Compute PRC values
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pos, pos_label = 1)

# Compute PRAUC
# 0.4307 for LogisticRegression
auc(recall, precision)

# Compute Brier score
# 0.194 for LogisticRegression
brier_score_loss(y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)


