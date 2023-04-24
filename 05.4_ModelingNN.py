# Used cars kicks classification - Modeling, NN with PyTorch Lightning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import optuna
import torch
import lightning.pytorch as pl
from sklearn.utils.class_weight import compute_class_weight
from XX_LightningClasses import TrainDataset, SeluDropoutModel, OptunaPruning


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
pl.seed_everything(1923, workers = True)


# Split train - validation data (10-fold CV takes too long)
x_train, x_val, y_train, y_val = train_test_split(
  x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
)


# Perform preprocessing
x_train = pipe_process.fit_transform(x_train, y_train)
x_val = pipe_process.transform(x_val)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
class_weight = torch.tensor(class_weight[1], dtype = torch.float32)


# Load data into TrainDataset
train_data = TrainhDataset(x_train, y_train)
val_data = TrainDataset(x_val, y_val)


# Create data loaders
train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 1024, num_workers = 0, shuffle = True)
val_loader = torch.utils.data.DataLoader(
      val_data, batch_size = 1024, num_workers = 0, shuffle = False)


# Define Optuna objective
def objective_nn(trial):
  
  # Define parameter ranges to tune over & suggest param set for trial
  n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
  hidden_size = trial.suggest_int("hidden_size", 2, 24, step = 2)
  learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log = True)
  l2 = trial.suggest_float("l2", 1e-4, 1e-3, log = True)
  dropout = trial.suggest_float("dropout", 0.01, 0.1, log = True)
  
  # Create hyperparameters dict
  hyperparams_dict = {
      "input_size": 88,
      "n_hidden_layers": n_hidden_layers,
      "hidden_size": hidden_size,
      "learning_rate": learning_rate,
      "l2": l2,
      "dropout": dropout,
      "class_weight": class_weight
    }
    
  # Create trainer & callbacks
  callback_earlystop = pl.callbacks.EarlyStopping(
      monitor = "val_loss",
      min_delta = 1e-4,
      patience = 10)
      
  callback_pruner = OptunaPruning(trial, monitor = "val_loss")

  trainer = pl.Trainer(
      max_epochs = 100,
      log_every_n_steps = 10, # The default is 50, but there are less training batches
      # than 50
      accelerator = "gpu", devices = "auto", precision = "16-mixed", 
      callbacks = [callback_earlystop, callback_pruner],
      logger = True,
      enable_progress_bar = False,
      enable_checkpointing = False
    )
    
  # Create & train model
  model = SeluDropoutModel(hyperparams_dict)
  trainer.fit(model, train_loader, val_loader)
    
  # Return best epoch's validation loss
  return trainer.callback_metrics["val_loss"].item()


# Create study
study_nn = optuna.create_study(
  sampler = optuna.samplers.TPESampler(seed = 1923),
  pruner = optuna.pruners.HyperbandPruner(),
  study_name = "tune_nn",
  direction = "minimize"
)


# Optimize study
study_nn.optimize(
  objective_nn, 
  n_trials = 250, 
  show_progress_bar = True)


# Retrieve and export trials
trials_nn = study_nn.trials_dataframe().sort_values("value", ascending = True)
trials_nn.to_csv("./ModifiedData/trials_nn2.csv", index = False)


# Import best trial
best_trial_nn = pd.read_csv("./ModifiedData/trials_nn2.csv").iloc[0,]


# Train & save NN model with best params, get best n. of epochs

# Define best hyperparameters
hyperparams_dict = {
      "input_size": 88,
      "n_hidden_layers": best_trial_nn["params_n_hidden_layers"],
      "hidden_size": best_trial_nn["params_hidden_size"],
      "learning_rate": best_trial_nn["params_learning_rate"],
      "l2": best_trial_nn["params_l2"],
      "dropout": best_trial_nn["params_dropout"],
      "class_weight": class_weight
    }
    
# Create trainer & callbacks
callback_earlystop = pl.callbacks.EarlyStopping(
      monitor = "val_loss",
      min_delta = 1e-4,
      patience = 10)

callback_checkpoint = pl.callbacks.ModelCheckpoint(
  monitor = "val_loss", save_last = True, save_top_k = 1,
  filename = "{epoch}-{val_loss:.4f}")
      
trainer = pl.Trainer(
      max_epochs = 100,
      log_every_n_steps = 10, # The default is 50, but there are less training batches
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

# Print best model path: epoch 32
trainer.checkpoint_callback.best_model_path
