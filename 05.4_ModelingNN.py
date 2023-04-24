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
from XX_LightningClasses import TorchDataset, SeluDropoutModel, OptunaPruning


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
pl.seed_everything(1923, workers = True)


# Get train-test indices (10 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define Optuna objective
def objective_nn(trial):
  
  # Define parameter ranges to tune over & suggest param set for trial
  n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
  hidden_size = trial.suggest_int("hidden_size", 16, 88, step = 8)
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2, log = True)
  l2 = trial.suggest_float("l2", 1e-5, 5e-2, log = True)
  dropout = trial.suggest_float("dropout", 0.05, 0.5)
  
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
    
    # Compute class weight
    classes = list(set(y_tr))
    class_weight = compute_class_weight("balanced", classes = classes, y = y_tr)
  
    # Create hyperparameters dict
    hyperparams_dict = {
      "input_size": 88,
      "n_hidden_layers": n_hidden_layers,
      "hidden_size": hidden_size,
      "learning_rate": learning_rate,
      "l2": l2,
      "dropout": dropout,
      "class_weight": torch.tensor(class_weight[1], dtype = torch.float32)
    }
    
    # Load data into TorchDataset
    train_data = TorchDataset(x_tr, y_tr)
    val_data = TorchDataset(x_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
      train_data, batch_size = 1024, num_workers = 0, shuffle = True)
    val_loader = torch.utils.data.DataLoader(
      val_data, batch_size = 1024, num_workers = 0, shuffle = False)
  
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
    
    # Save best epoch's validation loss to CV scores list
    cv_scores.append(trainer.callback_metrics["val_loss"].item())
  
  # Return avg. CV score of configuration  
  return np.mean(cv_scores)


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
  n_trials = 500, 
  show_progress_bar = True)


# Retrieve and export trials
trials_nn = study_nn.trials_dataframe().sort_values("value", ascending = True)
trials_nn.to_csv("./ModifiedData/trials_nn.csv", index = False)

