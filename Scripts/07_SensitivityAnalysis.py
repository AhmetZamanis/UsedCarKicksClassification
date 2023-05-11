# Used cars kicks classification - Sensitivity analysis
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/04_Preprocessing.py").read())


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import torch, torchvision
import lightning.pytorch as pl
from XX_LightningClasses import TrainDataset, TestDataset, SeluDropoutModel

from itertools import product


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
sample_weight = np.where(y_train == 1, class_weight[1], class_weight[0])


# Create dummy classifier which predicts the prior class probabilities
model_dummy = DummyClassifier(strategy = "prior")


# Train & predict with dummy classifier
model_dummy.fit(x_train, y_train)
preds_dummy = model_dummy.predict_proba(x_test)
preds_dummy = [x[1] for x in preds_dummy]


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


# Train & predict with logistic regression
pipe_logistic.fit(x_train, y_train, Logistic__sample_weight = sample_weight)
preds_logistic = pipe_logistic.predict_proba(x_test)
preds_logistic = [x[1] for x in preds_logistic]


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
  ("SVM", CalibratedClassifierCV(
    estimator = SGDClassifier(
      loss = "hinge",
      penalty = "elasticnet",
      alpha = best_trial_svm["params_reg_strength"],
      l1_ratio = best_trial_svm["params_l1_ratio"],
      max_iter = 13,
      n_iter_no_change = 1000, # Ensure model doesn't early stop based on train loss
      verbose = 1,
      random_state = 1923
      ),
    method = "isotonic"
    )
  )
])


# Train & predict with SVM
pipe_svm.fit(x_train, y_train, SVM__sample_weight = sample_weight)
preds_svm = pipe_svm.predict_proba(x_test)
preds_svm = [x[1] for x in preds_svm]


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


# Train & predict with XGBoost
pipe_xgb.fit(x_train, y_train, XGBoost__sample_weight = sample_weight)
preds_xgb = pipe_xgb.predict_proba(x_test)
preds_xgb = [x[1] for x in preds_xgb]


# Define & train NN model with best parameters
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
      enable_checkpointing = False
    )
    
# Train NN model
trainer.fit(model_nn, train_loader)

# Predict with NN model
y_prob = trainer.predict(model_nn, test_loader)
    
# Convert list of float16 Torch tensors to single float32 np.array
preds_nn = np.float32(y_prob[0].numpy().reshape(1, -1)[0])


# Retrieve prob. predictions, target labels, purchase prices in dataframes, sort in
# descending order according to prob. predictions
df_dummy = pd.DataFrame({
  "Price": x_test["VehBCost"],
  "Kick": y_test,
  "ProbKick": preds_dummy
})
df_dummy = df_dummy.sort_values("ProbKick", ascending = True)

df_logistic = pd.DataFrame({
  "Price": x_test["VehBCost"],
  "Kick": y_test,
  "ProbKick": preds_logistic
})
df_logistic = df_logistic.sort_values("ProbKick", ascending = True)

df_svm = pd.DataFrame({
  "Price": x_test["VehBCost"],
  "Kick": y_test,
  "ProbKick": preds_svm
})
df_svm = df_svm.sort_values("ProbKick", ascending = True)

df_xgb = pd.DataFrame({
  "Price": x_test["VehBCost"],
  "Kick": y_test,
  "ProbKick": preds_xgb
})
df_xgb = df_xgb.sort_values("ProbKick", ascending = True)

df_nn = pd.DataFrame({
  "Price": x_test["VehBCost"],
  "Kick": y_test,
  "ProbKick": preds_nn
})
df_nn = df_nn.sort_values("ProbKick", ascending = True)

# Define function to calculate profit-loss at given threshold prob. and number of
# cars to purchase
def calc_profit(threshold, num_purchases, df_probs):
  
  # Retrieve arrays of prices, labels, predicted probs
  price = df_probs["Price"].values
  kick = df_probs["Kick"].values
  prob = df_probs["ProbKick"].values
  
  # Retrieve n. of cars to purchase (if available at threshold)
  n = num_purchases
  
  # Get vector of purchase decision for the top N cars at threshold prob.
  decision = (prob[0:n] < threshold).astype(int)
  
  # Calculate profit/loss for each car, purchased or not
  profit = [((0.1 * price[i]) - (0.9 * price[i] * kick[i])) * decision[i] for i in range(n)]
  
  # Return n. of cars actually purchased, total profit / loss
  return sum(decision), sum(profit)


# Get combinations of thresholds - n. of cars to purchase
thresholds = np.arange(0, 1, 0.01)
num_buys = np.arange(0, len(y_test), 100)
thresholds_buys = list(product(thresholds, num_buys))


# Calculate n. of cars actually bought and profit at each threshold / buy combination
output_dummy = [calc_profit(x, y, df_dummy) for x, y in thresholds_buys]
decisions_dummy = [x[0] for x in output_dummy]
profits_dummy= [x[1] for x in output_dummy]

output_logistic = [calc_profit(x, y, df_logistic) for x, y in thresholds_buys]
decisions_logistic = [x[0] for x in output_logistic]
profits_logistic = [x[1] for x in output_logistic]

output_svm = [calc_profit(x, y, df_svm) for x, y in thresholds_buys]
decisions_svm = [x[0] for x in output_svm]
profits_svm = [x[1] for x in output_svm]

output_xgb = [calc_profit(x, y, df_xgb) for x, y in thresholds_buys]
decisions_xgb = [x[0] for x in output_xgb]
profits_xgb = [x[1] for x in output_xgb]

output_nn = [calc_profit(x, y, df_nn) for x, y in thresholds_buys]
decisions_nn = [x[0] for x in output_nn]
profits_nn = [x[1] for x in output_nn]


# Make long dataframes of threshold-purchase-profit values
df_long_dummy = pd.DataFrame({
  "Threshold": [x[0] for x in thresholds_buys],
  "Purchases": decisions_dummy,
  "Profits": profits_dummy,
  "Model": "Dummy"
})

df_long_logistic = pd.DataFrame({
  "Threshold": [x[0] for x in thresholds_buys],
  "Purchases": decisions_logistic,
  "Profits": profits_logistic,
  "Model": "Logistic"
})

df_long_svm = pd.DataFrame({
  "Threshold": [x[0] for x in thresholds_buys],
  "Purchases": decisions_svm,
  "Profits": profits_svm,
  "Model": "SVM"
})

df_long_xgb = pd.DataFrame({
  "Threshold": [x[0] for x in thresholds_buys],
  "Purchases": decisions_xgb,
  "Profits": profits_xgb,
  "Model": "XGBoost"
})

df_long_nn = pd.DataFrame({
  "Threshold": [x[0] for x in thresholds_buys],
  "Purchases": decisions_nn,
  "Profits": profits_nn,
  "Model": "Neural net"
})


# Combine long dataframes into one
df_long = pd.concat([df_long_dummy, df_long_logistic, df_long_svm, df_long_xgb, df_long_nn])

# Drop rows with duplicates of purchase-profit-model columns (cases where a 
# probability t results in buying n number of cars, and a higher t doesn't result 
# in more cars bought or a different profit value)
df_long = df_long.drop_duplicates(["Purchases", "Profits", "Model"])


# 2D lineplots of thresholds-purchases-profits
fig, ax = plt.subplots(3)
_ = fig.suptitle("Relationships between classifier threshold probability, number of cars purchased and profit")

_ = sns.lineplot(
  ax = ax[0],
  data = df_long, x = "Threshold", y = "Profits", hue = "Model")
_ = ax[0].set_xlabel("Threshold probability")
_ = ax[0].set_ylabel("Profits, $mil")
  

_ = sns.lineplot(
  ax = ax[1],
  data = df_long, x = "Threshold", y = "Purchases", hue = "Model", legend = False)
_ = ax[1].set_xlabel("Threshold probability (lowest value that results in N. cars purchased)")
_ = ax[1].set_ylabel("N. cars purchased")

_ = sns.lineplot(
  ax = ax[2],
  data = df_long, x = "Purchases", y = "Profits", hue = "Model", legend = False)
_ = ax[2].set_xlabel("N. cars purchased")
_ = ax[2].set_ylabel("Profits, $mil")


plt.show()
plt.savefig("./Plots/sensitivity.png", dpi = 300)
plt.close("all")


# Quasi-optimal combinations of threshold prob - n. purchases
optimal_dummy = df_long_dummy.loc[np.argmax(df_long_dummy["Profits"])]
optimal_logistic = df_long_logistic.loc[np.argmax(df_long_logistic["Profits"])]
optimal_svm = df_long_svm.loc[np.argmax(df_long_svm["Profits"])]
optimal_xgb = df_long_xgb.loc[np.argmax(df_long_xgb["Profits"])]
optimal_nn =df_long_nn.loc[np.argmax(df_long_nn["Profits"])]

df_optimal = pd.concat([
  optimal_dummy, optimal_logistic, optimal_svm, optimal_xgb, optimal_nn], 
  axis = 1).transpose()

df_optimal["Profits"] = df_optimal["Profits"] / 1e6
  
df_optimal = df_optimal.rename({
  "Threshold": "Threshold prob.",
  "Purchases": "N. cars purchased",
  "Profits": "Profits, $mil"
}, axis = 1)

df_optimal

