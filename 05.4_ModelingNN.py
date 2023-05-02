# Used cars kicks classification - Modeling, NN with PyTorch Lightning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
import torch, torchvision, torchmetrics
import lightning.pytorch as pl
from sklearn.utils.class_weight import compute_class_weight
from XX_LightningClasses import TrainDataset, SeluDropoutModel, OptunaPruning
import warnings


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
pl.seed_everything(1923, workers = True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# # Split train - validation data (10-fold CV takes too long)
# x_train, x_val, y_train, y_val = train_test_split(
#   x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
# )


# Get train-test indices (3 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define Optuna objective
def objective_nn(trial):
  
  # Define parameter ranges to tune over & suggest param set for trial
  n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 5)
  hidden_size = 2 ** trial.suggest_int("hidden_size", 1, 6)
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)
  l2 = trial.suggest_float("l2", 5e-5, 1e-2, log = True)
  dropout = trial.suggest_float("dropout", 1e-3, 0.1)
  loss_alpha = trial.suggest_float("loss_alpha", 0, 1, step = 0.1)
  loss_gamma = trial.suggest_float("loss_gamma", 0, 4, step = 0.25)
  
  # Create hyperparameters dict
  hyperparams_dict = {
      "input_size": 90,
      "n_hidden_layers": n_hidden_layers,
      "hidden_size": hidden_size,
      "learning_rate": learning_rate,
      "l2": l2,
      "dropout": dropout,
      "loss_alpha": loss_alpha,
      "loss_gamma": loss_gamma
    }
  
  # Crossvalidate the parameter set
  cv_scores = []
  
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
      val_data, batch_size = 1024, num_workers = 0, shuffle = False)
      
    # Create callbacks list
    callbacks = []
    
    callback_earlystop = pl.callbacks.EarlyStopping(
      monitor = "val_avg_precision", mode = "max",
      min_delta = 1e-4,
      patience = 10)
    callbacks.append(callback_earlystop)
    
    # Create Optuna pruner callback only for first CV fold
    if i == 0:
      callback_pruner = OptunaPruning(trial, monitor = "val_avg_precision")
      callbacks.append(callback_pruner)
    
    # Create trainer
    trainer = pl.Trainer(
      max_epochs = 100,
      log_every_n_steps = 5, # The default is 50, but there are less training batches
      # than 50
      accelerator = "gpu", devices = "auto", precision = "16-mixed", 
      callbacks = callbacks,
      enable_model_summary = False, # Disable model summary for tuning
      logger = True,
      enable_progress_bar = False,
      enable_checkpointing = False
    )
  
    # Create & train model
    model = SeluDropoutModel(hyperparams_dict)
    trainer.fit(model, train_loader, val_loader)
    
    # Append best epoch's validation average precision to CV scores list
    cv_scores.append(trainer.callback_metrics["val_avg_precision"].item())
    
  return np.mean(cv_scores)


# Create study
study_nn = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_nn",
  direction = "maximize"
)


# Optimize study
study_nn.optimize(
  objective_nn, 
  n_trials = 500, 
  show_progress_bar = True)


# Retrieve and export trials
trials_nn = study_nn.trials_dataframe().sort_values("value", ascending = False)
trials_nn.to_csv("./ModifiedData/trials_nn1.csv", index = False)


# Import best trial
best_trial_nn = pd.read_csv("./ModifiedData/trials_nn1.csv")
best_trial_nn = best_trial_nn.loc[
  best_trial_nn["state"] == "COMPLETE"].iloc[0,]


# Train & save NN model with best params, get best epoch
# Define best hyperparameters
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
    
# Create trainer & callbacks
callback_earlystop = pl.callbacks.EarlyStopping(
      monitor = "val_avg_precision", mode = "max",
      min_delta = 1e-4,
      patience = 10)

callback_checkpoint = pl.callbacks.ModelCheckpoint(
  monitor = "val_avg_precision", save_last = True, save_top_k = 1, mode = "max",
  filename = "{epoch}-{val_avg_precision:.4f}")
      
trainer = pl.Trainer(
      max_epochs = 100,
      log_every_n_steps = 5, # The default is 50, but there are less training batches
      # than 50
      accelerator = "gpu", devices = "auto", precision = "16-mixed", 
      callbacks = [callback_earlystop, callback_checkpoint],
      logger = True,
      enable_progress_bar = True,
      enable_checkpointing = True
    )








    
# Create & train model
model = SeluDropoutModel(hyperparams_dict)
trainer.fit(model, train_loader, val_loader)

# Best epoch: 17
trainer.checkpoint_callback.best_model_path
