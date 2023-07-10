
# LIGHTNING VALIDATION UNIT TEST
# Create hyperparameters dict
hyperparams_dict = {
      "input_size": 90,
      "n_hidden_layers": 2,
      "hidden_size": 16,
      "learning_rate": 0.5,
      "l2": 0.1,
      "dropout": 0.1,
      "loss_alpha": 0.25,
      "loss_gamma": 2
    }


# Store the CV scores for the parameter set
cv_scores = []
  
# Store the best n. of epochs
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

  # Load data into TrainDataset
  train_data = TrainDataset(x_tr, y_tr)
  val_data = TrainDataset(x_val, y_val)

  # Create data loaders
  train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 1024, num_workers = 0, shuffle = True)
  val_loader = torch.utils.data.DataLoader(
      val_data, batch_size = len(x_val), num_workers = 0, shuffle = False)
      
  # Create callbacks list
  callbacks = []
    
  # Create early stop callback
  callback_earlystop = pl.callbacks.EarlyStopping(
      monitor = "val_avg_precision", mode = "max",
      min_delta = 1e-4,
      patience = 10)
  callbacks.append(callback_earlystop)
    
  
    
  # Create trainer
  trainer = pl.Trainer(
      max_epochs = 500,
      log_every_n_steps = 5, # The default is 50, but there are less training batches
      # than 50
      accelerator = "gpu", devices = "auto", precision = "16-mixed", 
      callbacks = callbacks,
      enable_model_summary = False, 
      logger = True,
      enable_progress_bar = False, # Disable prog. bar, checkpoints for Optuna trials
      enable_checkpointing = False
    )
  
  # Create & train model
  model = SeluDropoutModel(hyperparams_dict = hyperparams_dict)
  trainer.fit(model, train_loader, val_loader)
    
  # Retrieve best val score and n. of epochs
  score = callbacks[0].best_score.cpu().numpy()
  epoch = trainer.current_epoch - callbacks[0].wait_count # Starts from 1
  cv_scores.append(score)
  best_epochs.append(epoch)
    
np.mean(cv_scores) 
np.median(best_epochs)
cv_scores

callbacks[0].best_score
callback_earlystop.best_score
trainer.current_epoch
callbacks[0].wait_count
callback_earlystop.wait_count
callback_earlystop.state_dict()




import time
# Test time of 1 calculation
start = time.time()
calc_profit(0.5, 5000, df_xgb)
end = time.time()
print(end - start)



import plotly.graph_objects as go
from scipy.interpolate import griddata

# 3D surface plot

# Original x-y-z values
x = df_profits["Threshold"].values
y = df_profits["PurchasesXGB"].values
z = df_profits["ProfitsXGB"].values

# x and y have to have same size and shape (n,)
xi = np.linspace(x.min(), x.max(), int((len(y) / 100)))
yi = np.linspace(y.min(), y.max(), int((len(y) / 100)))

# Create coordinate matrices from x and y
X,Y = np.meshgrid(xi, yi)

# Interpolate z values so z has shape (n, n)
Z = griddata((x, y), z, (X, Y), method = 'cubic')

fig = go.Figure(go.Surface(x=xi, y=yi, z=Z))
fig.show()




# Retrieve static inputs for Pulp model as arrays
pri = x_test["VehBCost"].values # Car prices
k = y_test.values # Kick or not
p_xgb = np.array(preds_xgb) # Predicted probs. XGB
p_nn = preds_nn # Predicted probs. NN
n = len(y_test) # Cars to buy

# Create Pulp model
m = LpProblem("OptimalThreshold", LpMaximize)


# Define variables
t = LpVariable("threshold_prob", 0, 1, "Continuous") # Threshold prob.
x = [LpVariable()] # List of binary purchase decision vars. for each car


# Add objective
m += lpSum(
  ((0.1 * pri[i]) - (0.9 * pri[i] * k[i])) * x for i in range(n)
  )


# Add constraints


# Optimize
print(m)

try:
  m.solve()
except:
  print(traceback.format_exc())


# Retrieve optimisation results, all values tried, plot results?
for var in m.variables():
  print(var.name, "=", var.varValue)









pulp.listSolvers(onlyAvailable=True)
pulp.pulpTestAll()
solver = pulpl.getSolver('GLPK_CMD')


Traceback (most recent call last):
  File "<string>", line 2, in <module>
  File "C:\Users\PC\DOCUME~1\Work\DATASC~1\GitHub\USEDCA~1\venv\lib\site-packages\pulp\pulp.py", line 1913, in solve
    status = solver.actualSolve(self, **kwargs)
  File "C:\Users\PC\DOCUME~1\Work\DATASC~1\GitHub\USEDCA~1\venv\lib\site-packages\pulp\apis\coin_api.py", line 137, in actualSolve
    return self.solve_CBC(lp, **kwargs)
  File "C:\Users\PC\DOCUME~1\Work\DATASC~1\GitHub\USEDCA~1\venv\lib\site-packages\pulp\apis\coin_api.py", line 202, in solve_CBC
    cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=subprocess.DEVNULL)
  File "C:\Program Files\Python310\lib\subprocess.py", line 837, in __init__
    errread, errwrite) = self._get_handles(stdin, stdout, stderr)
  File "C:\Program Files\Python310\lib\subprocess.py", line 1290, in _get_handles
    c2pwrite = _winapi.GetStdHandle(_winapi.STD_OUTPUT_HANDLE)
OSError: [WinError 6] The handle is invalid








from mip import *

# MIP optimization
n = model_xgb.add_var(
  name = "n_purchases", var_type = INTEGER, lb = 1, ub = len(y_test))
  
n = len(y_test)


# Create MIP model (choose solver, linear or non-linear?)
model_xgb = Model(
  name = "optimize_xgb",
  sense = MAXIMIZE, # Maximization problem
  solver_name = CBC # Solver algorithm
)


# Add variables: threshold prob, binary decision variable
t = model_xgb.add_var(
  name = "threshold_prob", var_type = CONTINUOUS, lb = 0, ub = 1)


# Add objective
model_xgb.objective = maximize(xsum(
    ((0.2 * price_xgb[i]) - (0.7 * price_xgb[i] * kick_xgb[i])) * (int((prob_xgb[i] <= t))) for i in range(n)
))


# Optimize
#model_xgb.max_gap = 0.05
status = model_xgb.optimize(max_seconds=300)
if status == OptimizationStatus.OPTIMAL:
  print('optimal solution cost {} found'.format(model_xgb.objective_value))
elif status == OptimizationStatus.FEASIBLE:
  print('sol.cost {} found, best possible: {}'.format(model_xgb.objective_value, model_xgb.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
  print('no feasible solution found, lower bound is: {}'.format(model_xgb.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
  print('solution:')
  for v in model_xgb.vars:
    print('{} : {}'.format(v.name, v.x))


# Retrieve optimisation results, all values tried, plot results?

# Add constraints: 
# 0 < t < 1, 
# 1 <= n_purchases <= len(y_test)












# Debug target encoder missing values
for i, (train_index, val_index) in enumerate(cv_indices):
  print(i)
  print(train_index)
  print(val_index)
  print("\n")



for i, (train_index, val_index) in enumerate(cv_indices):
  
    # Split training-validation data
    x_tr = x_train.iloc[train_index, ]
    y_tr = y_train.iloc[train_index, ]
    x_val = x_train.iloc[val_index, ]
    y_val = y_train.iloc[val_index, ]
    
    # Perform preprocessing
    x_tr = pipe_process.fit_transform(x_tr, y_tr)
    x_val = pipe_process.transform(x_val)
    
    # Return n. of missing values
    print(np.isnan(x_tr).sum())
    print(np.isnan(x_val).sum())
    print("\n")



# Debug SGD partial fit

# t_ does properly update after each epoch, and is reset with each validation fold.

# After looking at the source code, it seems fit and partial_fit use the "optimal"
# learning rate the same way, though it's hard to confirm. Both fit and partial_fit
# calls _fit_regressor which calls _plain_sgd, which seems to work the same way, 
# with only difference being max_iter = 1 for partial_fit.

# _plain_sgd does adjust the optimal learning rate and shuffle the data for each iter,
# regardless of number of iterations.
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
    
  # Define Logistic Regression classifier with SGD
  model_logistic = SGDClassifier(
    loss = "log_loss", # Log loss for probabilistic logistic regression
    penalty = "elasticnet",
    learning_rate = "optimal", # Dynamically adjusted based on regularization strength 
    random_state = 1923,
    verbose = 0, # Change to 1 to print epochs for debugging if needed
    alpha = 0.005,
    l1_ratio = 0.3
    )
    
  # Perform preprocessing
  x_tr = pipe_process.fit_transform(x_tr, y_tr)
  x_val = pipe_process.transform(x_val)
    
  # Perform epoch by epoch training with early stopping & pruning
  epoch_scores = []
  n_iter_no_change = 0
  tol = 0.001
    
  for epoch in range(1000):
    
    # Print epoch number, parameters that will be used, n. of weight updates performed
    print("Starting epoch: " + str(epoch) + "\n")
    
    print("Parameters: " + str(model_logistic.get_params()) + "\n")
    
    if epoch > 0:
      print("N. weight updates performed: " + str(model_logistic.t_) + "\n")
      
    # Train model for 1 epoch
    _ = model_logistic.partial_fit(x_tr, y_tr, classes = classes, sample_weight = sample_weight)
      
    # Score epoch
    y_pred = model_logistic.predict_proba(x_val)
    epoch_score = log_loss(y_val, y_pred, sample_weight = sample_weight_val)
      
    # Count epochs with no improvement after first 10 epochs
    if epoch > 9:
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
















from category_encoders.datasets import load_compass, load_postcodes

X, y = load_compass()
X, y = load_postcodes('binary')

# How to map zipcodes-states hierarchy

# With dict
dict_hierarchy = {
  "VNZIP1": {
    "FL": ("33619", "33..."),
    "GA": ("30212", "30..."),
    ...
    }
}

# All unique zipcodes for a state
df.loc[df["VNST"] == "FL", "VNZIP1"].unique().tolist()





test = pipe_preproc.fit_transform(x_train, y_train)
test[["Make", "BYRNO", "VNST", "VNZIP1"]].describe()


encode_target_zip.fit(x_train, y_train)
encode_target_zip.transform(x_train, y_train)

encode_target.fit(x_train, y_train)
encode_target.transform(x_train, y_train)


test = pipe_preproc.fit_transform(x_train, y_train)
test[0]
x_train.iloc[0,]



# Perform train-validation split
x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
  x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
)



# Test loop for XGB crossvalidation: Works.
best_scores_test = []

for split in cv_indices:

  # Get train-test indices and sets
  train_index = split[0]
  val_index = split[1]

  x_tr = x_train.iloc[train_index, ]
  y_tr = y_train.iloc[train_index, ]
  x_val = x_train.iloc[val_index, ]
  y_val = y_train.iloc[val_index, ]

  # Perform preprocessing
  x_tr = pipe_process.fit_transform(x_tr, y_tr)
  x_val = pipe_process.transform(x_val)

  # Fit model and print eval set score
  model_xgb.fit(X = x_tr, y = y_tr, eval_set = [(x_val, y_val)], verbose = True)
  best_scores_test.append(model_xgb.best_score)

del split



# Compute class weight (for labels 0, 1)
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight = np.where(y_train == 1, class_weight[1], class_weight[0])


# Compute class weight (for labels -1, 1)
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
sample_weight = np.where(y_train == 1, class_weight[0], class_weight[1])


# # Create logistic regression pipeline with optimal hyperparameters
# best_trial_logistic = pd.read_csv("./ModifiedData/trials_logistic.csv").iloc[0,]
# pipe_logistic = Pipeline(steps = [
#   ("preprocessing", pipe_process),
#   ("model_logistic", LogisticRegression(
#       penalty = "elasticnet",
#       C = (1 / best_trial_logistic["params_reg_strength"]),
#       l1_ratio = best_trial_logistic["params_l1_ratio"],
#       solver = "saga",
#       random_state = 1923,
#       max_iter = 1000,
#       class_weight = "balanced",
#     )
#   )
# ])


# Fit on training data, predict testing data, retrieve positive label probs
pipe_logistic.fit(x_train, y_train)
y_pred = pipe_logistic.predict(x_test)
y_prob = pipe_logistic.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])

# Compute PRC values
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pos, pos_label = 1)

pd.DataFrame({
  "Model": "Logistic",
  "Precision": precision,
  "Recall": recall})
  
for key in models_dict.keys():
  print(models_dict[key])
  

# Compute PRAUC
# 0.4307 for LogisticRegression
# 0.4223 for SGD with native early stopping
auc(recall, precision)

# Compute Brier score
# 0.194 for LogisticRegression
# 0.1958 for SGD with native early stopping
brier_score_loss(y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)


# Compute Brier skill score for each model, with dummy classifier as reference:
# 1 - (brier_score_model / brier_score_dummy) (1 is best, 0 is worst)





# Make named dictionary of models
models_dict = {
  "Dummy": pipe_dummy,
  "Logistic": pipe_logistic,
  "SVM": pipe_svm,
  "XGBoost": pipe_xgb
}


# Fit models




# Define function that fits every model pipeline & scores test data, plots PRC
# curve
def score_models(models_dict):
  
  # # Make dataframe to store precision-recall values for each threshold
  # df_prc = pd.DataFrame(
  #   columns = ["Model", "Precision", "Recall"])
  
  # Make dataframe to store PRAUC & Brier scores
  df_scores = pd.DataFrame(
    columns = ["Model", "Avg. precision score", "Brier score"]
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
      
    # Compute Average precision score
    avg_precision = average_precision_score(y_test, y_prob_pos, pos_label = 1)
    
    # Compute Brier score
    brier_score = brier_score_loss(
      y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)
    
    # Make dataframe of precision-recall values for each threshold
    prc = pd.DataFrame({
      "Model": key,
      "Precision": precision,
      "Recall": recall
    })
    
    # Concatenate PRC values to full dataframe
    df_prc = pd.concat([df_prc, prc])
    
    # Make dataframe of PRAUC & Brier scores, concatenate to full dataframe
    scores = pd.DataFrame({
      "Model": [key],
      "Avg. precision score": [avg_precision],
      "Brier score": [brier_score]
    })
    df_scores = pd.concat([df_scores, scores])
  
  # Drop dummy classifier's PRC values
  df_prc = df_prc.loc[df_prc["Model"] != "Dummy"]
  
  return df_prc, df_scores


# Retrieve PRC values & performance scores
df_prc, df_scores = score_models(models_dict)


# Plot PRC curves
_ = sns.lineplot(x = "Recall", y = "Precision", data = df_prc, hue = "Model")
plt.show()
plt.close("all")




# Fit on training data, predict testing data, retrieve positive label probs
pipe_dummy.fit(x_train, y_train)
y_pred = pipe_dummy.predict(x_test)
y_prob = pipe_dummy.predict_proba(x_test)
y_prob_pos = np.array([x[1] for x in y_prob])

# Compute PRC values
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_pos, pos_label = 1)


# Compute Average Precision
# 0.4307 for LogisticRegression
# 0.4223 for SGD with native early stopping
average_precision_score(y_test, y_prob_pos, pos_label = 1)


# Compute Brier score
# 0.194 for LogisticRegression
# 0.1958 for SGD with native early stopping
brier_score_loss(y_test, y_prob_pos, pos_label = 1, sample_weight = sample_weight_test)


# Compute Brier skill score for each model, with dummy classifier as reference:
# 1 - (brier_score_model / brier_score_dummy) (1 is best, 0 is worst)




# Create dummy classifier pipeline
pipe_dummy = Pipeline(steps = [
  ("preprocessing", pipe_process),
  ("model_dummy", DummyClassifier(strategy = "prior"))
])




# loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weight, dtype = torch.float32), reduction = "mean")
#y = y.unsqueeze(1).float()

# One iteration of dataloader
train_x, train_y = next(iter(train_loader))
