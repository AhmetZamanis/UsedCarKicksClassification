# Used cars kicks classification - Model scoring
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/04_Preprocessing.py").read())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import torch, torchvision
import lightning.pytorch as pl
from XX_LightningClasses import TrainDataset, TestDataset, SeluDropoutModel


# Set plotting options
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.autolayout"] = True
sns.set_style("darkgrid")


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
pl.seed_everything(1923, workers = True)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight_train = np.where(y_train == 1, class_weight[1], class_weight[0])
sample_weight_test = np.where(y_test == 1, class_weight[1], class_weight[0])


# Create dummy classifier which predicts the prior class probabilities
model_dummy = DummyClassifier(strategy = "prior")


# Create logistic regression pipeline with optimal hyperparameters
best_trial_logistic = pd.read_csv("./ModifiedData/trials_logistic.csv").iloc[0,]
pipe_logistic = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("Logistic", SGDClassifier(
      loss = "log_loss",
      penalty = "elasticnet",
      alpha = best_trial_logistic["params_reg_strength"],
      l1_ratio = best_trial_logistic["params_l1_ratio"],
      max_iter = 26,
      n_iter_no_change = 1000, # Ensure model doesn't early stop based on train loss
      verbose = 1,
      random_state = 1923
    )
  )
])


# Create SVM pipeline with optimal hyperparameters
best_trial_svm = pd.read_csv("./ModifiedData/trials_svm.csv").iloc[0,]
pipe_svm = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("KernelTrick", RBFSampler(
      gamma = "scale",
      n_components = 100,
      random_state = 1923
    )
  ),
  ("SVM", CalibratedClassifierCV(SGDClassifier(
      loss = "hinge",
      penalty = "elasticnet",
      alpha = best_trial_svm["params_reg_strength"],
      l1_ratio = best_trial_svm["params_l1_ratio"],
      max_iter = 13,
      n_iter_no_change = 1000, # Ensure model doesn't early stop based on train loss
      verbose = 1,
      random_state = 1923
      )
    )
  )
])


# Create XGBoost pipeline with optimal hyperparameters
best_trial_xgb = pd.read_csv("./ModifiedData/trials_xgb.csv").iloc[0,]
pipe_xgb = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("XGBoost", XGBClassifier(
      objective = "binary:logistic",
      n_estimators = 20,
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


# Define NN model with best parameters
best_trial_nn = pd.read_csv("./ModifiedData/trials_nn.csv").iloc[0,]
hyperparams_dict = {
      "input_size": 90,
      "n_hidden_layers": best_trial_nn["params_n_hidden_layers"],
      "hidden_size": 2 ** best_trial_nn["params_hidden_size"],
      "learning_rate": best_trial_nn["params_learning_rate"],
      "l2": best_trial_nn["params_l2"],
      "dropout": best_trial_nn["params_dropout"],
      "loss_alpha": best_trial_nn["params_loss_alpha"],
      "loss_gamma": best_trial_nn["params_loss_gamma"]
    }
model_nn = SeluDropoutModel(hyperparams_dict)


# Make dict of models
models_dict = {
  "Dummy": model_dummy,
  "Logistic": pipe_logistic,
  "SVM": pipe_svm,
  "XGBoost": pipe_xgb,
  "Neural net": model_nn
}


# Train & predict each model
preds_dict = {}

for key in models_dict.keys():
  
  # Retrieve model
  model = models_dict[key]
  
  # Fit model
  if key == "Dummy":
    model.fit(x_train, y_train)
  
  elif key == "Neural net":
    
    # Apply scikit preprocessing pipeline
    x_tr = pipe_process.fit_transform(x_train, y_train)
    x_test1 = pipe_process.transform(x_test)
    
    # Create train & test Datasets, dataloaders
    train_data = TrainDataset(x_tr, y_train)
    test_data = TestDataset(x_test1)
    
    train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 1024, num_workers = 0, shuffle = True)
    test_loader = torch.utils.data.DataLoader(
      test_data, batch_size = len(test_data), num_workers = 0, shuffle = False)
      
    # Create trainer
    trainer = pl.Trainer(
      max_epochs = 9, # Best epoch from ModelingNN
      log_every_n_steps = 5, # The default is 50, but there are less training batches
      # than 50
      accelerator = "gpu", devices = "auto", precision = "16-mixed", 
      logger = True,
      enable_progress_bar = True,
      enable_checkpointing = True
    )
    
    # Train model
    trainer.fit(model, train_loader)

  else:
    
    # Create unique sample weights argument for pipeline.fit
    kwargs = {model.steps[-1][0] + "__sample_weight": sample_weight_train}
    
    # Fit pipeline
    model.fit(x_train, y_train, **kwargs)
  
  # Predict positive class prob
  if key == "Neural net":
    y_prob = trainer.predict(model_nn, test_loader)
    
    # Convert a list of float16 Torch tensors to single float32 np.array
    preds_dict[key] = np.float32(y_prob[0].numpy().reshape(1, -1)[0])
  
  else:  
    y_prob = model.predict_proba(x_test)
    y_prob = np.array([x[1] for x in y_prob])
    preds_dict[key] = y_prob
  

# Retrieve AP & Brier scores (weighted) for each model
scores_avg_precision = {}
scores_brier = {}

for key in preds_dict.keys():

  # Retrieve average precision scores
  avg_precision = average_precision_score(y_test, preds_dict[key])
  scores_avg_precision[key] = avg_precision
  
  # Retrieve Brier scores
  brier_score = brier_score_loss(
    y_test, preds_dict[key], sample_weight = sample_weight_test)
  scores_brier[key] = brier_score
  
 
# Retrieve Brier skill scores for each model, with dummy classifier as reference
scores_brier_skill = {}

for key in preds_dict.keys():
  
  brier_skill = 1 - (scores_brier[key] / scores_brier["Dummy"])
  scores_brier_skill[key] = brier_skill
  

# Retrieve F1 scores at different thresholds
scores_f1 = {}
scores_f1_best = {}

scores_precision = {}
scores_precision_best = {}

scores_recall = {}
scores_recall_best = {}

threshold_probs = {}
threshold_probs_best = {}

for key in preds_dict.keys():
  
  # Retrieve the precision & recall pairs at various thresholds, calculate F1
  # scores from them
  precision, recall, thresholds = precision_recall_curve(y_test, preds_dict[key])
  f1_scores = 2 * recall * precision / (recall + precision)
    
  scores_f1[key] = f1_scores
  scores_f1_best[key] = max(f1_scores)
  
  scores_precision[key] = precision
  scores_precision_best[key] = precision[np.argmax(f1_scores)]
  
  scores_recall[key] = recall
  scores_recall_best[key] = recall[np.argmax(f1_scores)]
  
  threshold_probs[key] = thresholds
  threshold_probs_best[key] = thresholds[np.argmax(f1_scores)]
  
  
# Retrieve dataframe of scores
df_scores = pd.DataFrame(
  {
  "Avg. precision (PRAUC)": scores_avg_precision.values(),
  "Brier score (class weighted)": scores_brier.values(),
  "Brier skill score (class weighted)": scores_brier_skill.values(),
  "Best F1 score": scores_f1_best.values(),
  "Precision at best F1": scores_precision_best.values(),
  "Recall at best F1": scores_recall_best.values(),
  "Threshold prob. at best F1": threshold_probs_best.values()
  }, index = preds_dict.keys()
)
df_scores.to_csv("./ModifiedData/scores.csv", index = True)


# Get dataframes for F1 score - threshold prob plots
# Logistic
df_f1_logistic = pd.DataFrame(
  {"F1 score": scores_f1["Logistic"][:-1], # N. scores = N. thresholds + 1
   "Precision": scores_precision["Logistic"][:-1],
   "Recall": scores_recall["Logistic"][:-1],
   "Threshold prob.": threshold_probs["Logistic"]
  }
).melt(
  value_vars = ["F1 score", "Precision", "Recall"], 
  var_name = "Metric",
  value_name = "Score",
  id_vars = "Threshold prob."
  )

# SVM
df_f1_svm = pd.DataFrame(
  {"F1 score": scores_f1["SVM"][:-1],
   "Precision": scores_precision["SVM"][:-1],
   "Recall": scores_recall["SVM"][:-1],
   "Threshold prob.": threshold_probs["SVM"]
  }
).melt(
  value_vars = ["F1 score", "Precision", "Recall"], 
  var_name = "Metric",
  value_name = "Score",
  id_vars = "Threshold prob."
  )

# XGBoost
df_f1_xgb = pd.DataFrame(
  {"F1 score": scores_f1["XGBoost"][:-1],
   "Precision": scores_precision["XGBoost"][:-1],
   "Recall": scores_recall["XGBoost"][:-1],
   "Threshold prob.": threshold_probs["XGBoost"]
  }
).melt(
  value_vars = ["F1 score", "Precision", "Recall"], 
  var_name = "Metric",
  value_name = "Score",
  id_vars = "Threshold prob."
  )

# NN
df_f1_nn = pd.DataFrame(
  {"F1 score": scores_f1["Neural net"][:-1],
   "Precision": scores_precision["Neural net"][:-1],
   "Recall": scores_recall["Neural net"][:-1],
   "Threshold prob.": threshold_probs["Neural net"]
  }
).melt(
  value_vars = ["F1 score", "Precision", "Recall"], 
  var_name = "Metric",
  value_name = "Score",
  id_vars = "Threshold prob."
  )


# Get dataframes for stacked histogram plots
# Logistic
df_preds_logistic = pd.DataFrame({
  "Prob. predictions": preds_dict["Logistic"],
  "Actual labels": y_test,
})

# SVM
df_preds_svm = pd.DataFrame({
  "Prob. predictions": preds_dict["SVM"],
  "Actual labels": y_test,
})

# XGBoost
df_preds_xgb = pd.DataFrame({
  "Prob. predictions": preds_dict["XGBoost"],
  "Actual labels": y_test,
})

# NN
df_preds_nn = pd.DataFrame({
  "Prob. predictions": preds_dict["Neural net"],
  "Actual labels": y_test,
})


# Plot precision-recall curves
fig, ax = plt.subplots()
for key in preds_dict.keys():
  _ = PrecisionRecallDisplay.from_predictions(y_test, preds_dict[key], name = key, ax = ax)
_ = plt.title("Precision-recall curves of classifiers")
_ = plt.legend(loc = "upper right")
plt.show()
plt.savefig("./Plots/prc.png", dpi = 300)
plt.close("all")


# Plot F1 score - threshold prob. plots
fig, ax = plt.subplots(4, sharex = True, sharey = True)
_ = fig.suptitle("F1 - precision - recall scores across threshold probabilities")

# Logistic
_ = sns.lineplot(
  ax = ax[0], 
  x = "Threshold prob.", y = "Score", hue = "Metric", 
  data = df_f1_logistic)
_ = ax[0].set_title("Logistic")

# SVM
_ = sns.lineplot(
  ax = ax[1], 
  x = "Threshold prob.", y = "Score", hue = "Metric", 
  data = df_f1_svm, legend = False)
_ = ax[1].set_title("SVM")

# XGBoost
_ = sns.lineplot(
  ax = ax[2], 
  x = "Threshold prob.", y = "Score", hue = "Metric", 
  data = df_f1_xgb, legend = False)
_ = ax[2].set_title("XGBoost")

# NN
_ = sns.lineplot(
  ax = ax[3], 
  x = "Threshold prob.", y = "Score", hue = "Metric", 
  data = df_f1_nn, legend = False)
_ = ax[3].set_title("Neural net")

plt.show()
plt.savefig("./Plots/f1.png", dpi = 300)
plt.close("all")


# Plot predicted probability distributions of classifiers
fig, ax = plt.subplots(4, sharex = True, sharey = True)
_ = fig.suptitle("Distributions of positive class probability predictions")

# Logistic
_ = sns.histplot(
  ax = ax[0], 
  x = "Prob. predictions", 
  hue = "Actual labels",
  multiple = "stack",
  data = df_preds_logistic)
_ = ax[0].set_title("Logistic")
_ = ax[0].set_ylabel("N. of times predicted")

# SVM
_ = sns.histplot(
  ax = ax[1], 
  x = "Prob. predictions",
  hue = "Actual labels",
  multiple = "stack",
  data = df_preds_svm,
  legend = False)
_ = ax[1].set_title("SVM")
_ = ax[1].set_ylabel("N. of times predicted")

# XGBoost
_ = sns.histplot(
  ax = ax[2], 
  x = "Prob. predictions",
  hue = "Actual labels",
  multiple = "stack",
  data = df_preds_xgb,
  legend = False)
_ = ax[2].set_title("XGBoost")
_ = ax[2].set_ylabel("N. of times predicted")

# NN
_ = sns.histplot(
  ax = ax[3], 
  x = "Prob. predictions",
  hue = "Actual labels",
  multiple = "stack",
  data = df_preds_nn,
  legend = False)
_ = ax[3].set_title("Neural net")
_ = ax[3].set_xlabel("Probability predictions for positive class")
_ = ax[3].set_ylabel("N. of times predicted")

plt.show()
plt.savefig("./Plots/prob_dist.png", dpi = 300)
plt.close("all")
