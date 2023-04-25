# Used cars kicks classification - NN implementation in PyTorch Lightning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


import torch
import lightning.pytorch as pl

# Careful: Optuna still uses "pytorch_lightning" to import the Pruning integration
# with Ligtning, but Lightning itself is now imported as lightning.pytorch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# Define TrainDataset class: Takes in preprocessed features & targets
class TrainDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features & targets
  def __init__(self, x_train, y_train): 
    self.x = torch.tensor(x_train, dtype = torch.float32) # Store features
    self.y = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1) # Store targets
  
  # Return data length  
  def __len__(self):
    return len(self.x) 
  
  # Return a pair of features & target
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


# Define TestDataset class: Takes in preprocessed features only
class TestDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features 
  def __init__(self, x_test): 
    self.x = torch.tensor(x_test, dtype = torch.float32) # Store features
  
  # Return data length  
  def __len__(self):
    return len(self.x) 
  
  # Return one set of features
  def __getitem__(self, idx):
    return self.x[idx]

 
# Define Lightning module
class SeluDropoutModel(pl.LightningModule):
  
  # Initialize model
  def __init__(self, hyperparams_dict):
    
    # Delegate function to parent class
    super().__init__() 
    self.save_hyperparameters(logger = False) # Save external hyperparameters so
    # they are available when loading saved models
  
    # Define hyperparameters
    self.n_hidden_layers = hyperparams_dict["n_hidden_layers"]
    self.input_size = hyperparams_dict["input_size"]
    self.hidden_size = hyperparams_dict["hidden_size"]
    self.learning_rate = hyperparams_dict["learning_rate"]
    self.l2 = hyperparams_dict["l2"]
    self.dropout = hyperparams_dict["dropout"]
    self.class_weight = hyperparams_dict["class_weight"]
    
    # Define architecture 
    # Initialize layers list with first hidden layer
    self.layers_list = torch.nn.ModuleList([
      torch.nn.Linear(self.input_size, self.hidden_size), # Hidden layer 1
      torch.nn.SELU(), # Activation 1
      torch.nn.AlphaDropout(self.dropout) # Dropout 1
      ])
    
    # Append extra hidden layers to layers list, according to hyperparameter
    for n in range(0, (self.n_hidden_layers - 1)):
      self.layers_list.extend([
        torch.nn.Linear(self.hidden_size, self.hidden_size), # Hidden layer N
        torch.nn.SELU(), # Activation N
        torch.nn.AlphaDropout(self.dropout) # Dropout N
      ])
    
    # Append output layer to layers list
    self.layers_list.append(
      torch.nn.Linear(self.hidden_size, 1) # Output layer
      # No sigmoid activation here, because the loss function has that built-in
      )
      
    # Define full network from layers list
    self.network = torch.nn.Sequential(*self.layers_list)
    
    # Sigmoid activation for prediction step only, not part of forward propagation
    self.sigmoid = torch.nn.Sequential(torch.nn.Sigmoid())
      
    # Initialize weights to conform with self-normalizing SELU activation
    for layer in self.network:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    
  # Define forward propagation
  def forward(self, x):
    output = self.network(x.view(x.size(0), -1))
    return output # Returns logits, not probabilities
  
  # Define training loop
  def training_step(self, batch, batch_idx):
    
    # Perform training, calculate log & return loss
    x, y = batch
    output = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      output, y, pos_weight = self.class_weight) # Loss function applies sigmoid
      # activation before calculating loss
    self.log(
      "train_loss", loss, 
      on_step = False, on_epoch = True, prog_bar = True, logger = True)
    return loss
  
  # Define validation loop
  def validation_step(self, batch, batch_idx):
    
    # Perform validation, calculate log & return loss
    x, y = batch
    output = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      output, y, pos_weight = self.class_weight)
    self.log(
      "val_loss", loss, 
      on_step = False, on_epoch = True, prog_bar = True, logger = True)
    return loss
  
  # Define prediction method (because the default just runs forward(), which
  # doesn't have sigmoid activation and doesn't return probabilities)
  def predict_step(self, batch, batch_idx):
    
    # Run the forward propagation, apply sigmoid activation to return probs.
    x = batch 
    output = self.forward(x)
    pred = self.sigmoid(output)
    return pred
    
  # Define optimization algorithm, LR scheduler
  def configure_optimizers(self):
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.Adam(
      self.parameters(), lr = self.learning_rate, weight_decay = self.l2)
    
    # Cyclic LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
      optimizer, 
      base_lr = self.learning_rate, max_lr = (self.learning_rate * 5), 
      step_size_up = 200, # Heuristic: (2-8 * steps (batches) in one epoch)
      cycle_momentum = False, # Not compatible with Adam optimizer 
      mode = "exp_range", gamma = 0.99995)
    
    return {
    "optimizer": optimizer,
    "lr_scheduler": {
      "scheduler": lr_scheduler,
      "interval": "step",
      "frequency": 1
      }
    }


# Create copy of Optuna callback with lightning.pytorch namespace as a workaround,
# as Optuna code uses pytorch_lightning namespace which causes an error
class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

