# Used cars kicks classification - Optimization
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import plotly.express as px
from sklearn.utils.class_weight import compute_class_weight
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
    
# Train model
trainer.fit(model_nn, train_loader)

# Predict with NN model
y_prob = trainer.predict(model_nn, test_loader)
    
# Convert list of float16 Torch tensors to single float32 np.array
preds_nn = np.float32(y_prob[0].numpy().reshape(1, -1)[0])


# Retrieve prob. predictions, target labels, purchase prices in dataframes, sort in
# descending order according to prob. predictions
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
output_xgb = [calc_profit(x, y, df_xgb) for x, y in thresholds_buys]
decisions_xgb = [x[0] for x in output_xgb]
profits_xgb = [x[1] for x in output_xgb]

output_nn = [calc_profit(x, y, df_nn) for x, y in thresholds_buys]
decisions_nn = [x[0] for x in output_nn]
profits_nn = [x[1] for x in output_nn]


# Make long dataframe of threshold-purchase-profit values
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

df_long = pd.concat([df_long_xgb, df_long_nn])

# Drop rows with duplicates of purchase-profit-model columns (cases where a 
# probability t results in buying n number of cars, and a higher t doesn't result 
# in more cars bought or a different profit value)
df_long = df_long.drop_duplicates(["Purchases", "Profits", "Model"])


# 3D scatterplot of thresholds-profits
fig = px.line_3d(
  df_long, x = 'Threshold', y = 'Purchases', z = 'Profits', 
  color = 'Model', line_dash = "Model", markers = True,
  title = "Sensitivity analysis: Profits / losses from purchasing top N cars predicted least likely to be kicks",
  labels = {
    "Threshold": "Threshold prob. of classifier",
    "Purchases": "Number of purchases",
    "Profits": "Total profit, $"
  })
fig.show()


# Quasi-optimal combination of threshold prob - n. purchases
df_long_xgb.loc[np.argmax(df_long_xgb["Profits"])]
df_long_nn.loc[np.argmax(df_long_nn["Profits"])]
 


