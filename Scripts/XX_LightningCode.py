# Used cars kicks classification - NN implementation in PyTorch Lightning
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import torch
import lightning.pytorch as pl
from sklearn.utils.class_weight import compute_class_weight


# Set Torch settings
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
pl.seed_everything(1923, workers = True)


# Apply scikit preprocessing pipeline
x_train = pipe_process.fit_transform(x_train, y_train)
x_test = pipe_process.transform(x_test)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)


# Set hyperparameters
hyperparams_dict = {
  "n_hidden_layers": 2,
  "input_size": 88,
  "hidden_size": 64,
  "learning_rate": 1e-3,
  "l2": 1e-5,
  "dropout": 0.1,
  "class_weight": torch.tensor(class_weight[1], dtype = torch.float32)
}


# Define Dataset class: Takes in preprocessed features & targets
class TorchDataset(torch.utils.data.Dataset):
  
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
 

# Define Lightning module
class SeluDropoutModel(pl.LightningModule):
  
  # Initialize model
  def __init__(self, hyperparams_dict):
    
    # Delegate function to parent class
    super().__init__() 
    self.save_hyperparameters(logger = False)
  
    
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
    
    # Append extra hidden layers to layers list
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
      
    # Full network
    self.network = torch.nn.Sequential(*self.layers_list)
      
    # Initialize weights
    for layer in self.network:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity = "linear")
        torch.nn.init.zeros_(layer.bias)
    
  # Define forward propagation
  def forward(self, x):
    output = self.network(x.view(x.size(0), -1))
    return output
  
  # Define training loop
  def training_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return loss
    x, y = batch
    output = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      output, y, pos_weight = self.class_weight)
    self.log(
      "train_loss", loss, 
      on_epoch = True, prog_bar = True, logger = True)
    return loss
  
  # Define validation loop
  def validation_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return loss
    x, y = batch
    output = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      output, y, pos_weight = self.class_weight)
    self.log(
      "val_loss", loss, 
      on_epoch = True, prog_bar = True, logger = True)
    return loss
  
  # Define prediction method (because the default just runs forward(), which
  # doesn't have sigmoid activation without the loss function)
  def predict_step(self, batch, batch_idx):
    
    # Run the forward propagation, apply sigmoid activation
    return torch.nn.Sigmoid(self.network(x.view(x.size(0), -1)))
    
  # Define optimization algorithm, LR scheduler
  def configure_optimizers(self):
    
    # Optimizer
    optimizer = torch.optim.Adam(
      self.parameters(), lr = self.learning_rate, weight_decay = self.l2)
    
    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
      optimizer, 
      base_lr = self.learning_rate, max_lr = (self.learning_rate * 5), 
      step_size_up = 200, # Heuristic: (2-8 * steps in one epoch)
      cycle_momentum = False, mode = "exp_range", gamma = 0.99995)
    
    return {
    "optimizer": optimizer,
    "lr_scheduler": {
      "scheduler": lr_scheduler,
      "interval": "step",
      "frequency": 1
      }
    }


# Load data
train_data = TorchDataset(x_train, y_train)
val_data = TorchDataset(x_test, y_test)


# Create data loaders
train_loader = torch.utils.data.DataLoader(
  train_data, batch_size = 1024, num_workers = 0, shuffle = True)
val_loader = torch.utils.data.DataLoader(
  val_data, batch_size = 1024, num_workers = 0, shuffle = False)


# Create trainer & callbacks
callback_earlystop = pl.callbacks.EarlyStopping(
    monitor = "val_loss",
    min_delta = 1e-3,
    patience = 10)

callback_checkpoint = pl.callbacks.ModelCheckpoint(
  monitor = "val_loss", save_last = True, save_top_k = 1,
  filename = "{epoch}-{val_loss:.4f}")
    
trainer = pl.Trainer(
  max_epochs = 100,
  accelerator = "gpu", precision = "16-mixed", 
  callbacks = [callback_earlystop, callback_checkpoint],
  deterministic = True, enable_progress_bar = True, logger = True,
  enable_checkpointing = True
  )


# Train model
model = SeluDropoutModel(hyperparams_dict = hyperparams_dict)
trainer.fit(model, train_loader, val_loader)


# Retrieve val score of best epoch, right after training
trainer.callback_metrics["val_loss"].item()


# Access path of best checkpoint, right after training
best_path = trainer.checkpoint_callback.best_model_path


# Load best checkpoint without training (some training info like current_epoch
# is stored in trainer, not model state)
model = SeluDropoutModel.load_from_checkpoint(best_path)


# Continue training from best checkpoint
trainer.fit(SeluDropoutModel(), ckpt_path = best_path)


# Predict with best checkpoint
trainer.predict(SeluDropoutModel(), test_loader, ckpt_path = best_path)
