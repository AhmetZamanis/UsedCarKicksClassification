# Used cars kicks classification - Modeling, NN with PyTorch Lightning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("./Scripts/04_Preprocessing.py").read())


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


# # Split train - validation data (if 3-fold CV takes too long)
# x_train, x_val, y_train, y_val = train_test_split(
#   x_train, y_train, test_size = 0.2, random_state = 1923, stratify = y_train
# )


# Get train-test indices (3 pairs)
cv_indices = list(cv_kfold.split(x_train, y_train))


# Define validation function
def validate_nn(hyperparams_dict, trial, tol = 1e-4):
  
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
      min_delta = tol,
      patience = 10)
    callbacks.append(callback_earlystop)
    
    # Create Optuna pruner callback only for first CV fold
    if i == 0:
      callback_pruner = OptunaPruning(trial, monitor = "val_avg_precision")
      callbacks.append(callback_pruner)
    
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
    
  # Return the mean CV score, median best epoch
  return np.mean(cv_scores), np.median(best_epochs)
  

# Define Optuna objective
def objective_nn(trial):
  
  # Define parameter ranges to tune over & suggest param set for trial
  n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
  hidden_size = 2 ** trial.suggest_int("hidden_size", 1, 6)
  learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)
  l2 = trial.suggest_float("l2", 0, 1e-2)
  dropout = trial.suggest_float("dropout", 1e-3, 0.25)
  loss_alpha = trial.suggest_float("loss_alpha", 0, 1)
  loss_gamma = trial.suggest_float("loss_gamma", 0, 4)
  
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
    
  # Validate hyperparameter set
  score, epoch = validate_nn(hyperparams_dict = hyperparams_dict, trial = trial)
  
  # Report best n. of epochs
  trial.set_user_attr("n_epochs", epoch)
  
  return score
  

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
trials_nn.to_csv("./ModifiedData/trials_nn.csv", index = False)

