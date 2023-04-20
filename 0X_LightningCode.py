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


# Apply scikit preprocessing pipeline
x_train = pipe_process.fit_transform(x_train, y_train)
x_test = pipe_process.transform(x_test)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
class_weight = torch.tensor(class_weight[1], dtype = torch.float32)


# Set hyperparameters
hidden_size = 64
learning_rate = 1e-3


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
class TwoHiddenLayers(pl.LightningModule):
  
  # Initialize model
  def __init__(self):
    
    # Delegate function to parent class
    super().__init__() 
    
    # Define architecture  
    self.network = torch.nn.Sequential(
    torch.nn.Linear(in_features = 88, out_features = hidden_size), # Input layer
    torch.nn.ReLU(), # Activation 1
    torch.nn.Linear(in_features = hidden_size, out_features = hidden_size), # Hidden 1
    torch.nn.ReLU(), # Activation 2
    torch.nn.Linear(in_features = hidden_size,  out_features = hidden_size), # Hidden 2
    torch.nn.Linear(in_features = hidden_size, out_features = 1) # Output layer
      # No Sigmoid activation here because the loss function has it built-in
    )
  
  # Define forward propagation
  def forward(self, x):
    output = self.network(x.view(x.size(0), -1))
    return output
  
  # Define training loop
  def training_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return loss
    x, y = batch
    pred = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      pred, y, pos_weight = class_weight)
    self.log(
      "train_loss", loss, 
      on_step = True, on_epoch = True, prog_bar = True, logger = True)
    return loss
  
  # Define validation loop
  def validation_step(self, batch, batch_idx):
    
    # Perform training, calculate, log & return loss
    x, y = batch
    pred = self.forward(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
      pred, y, pos_weight = class_weight)
    self.log(
      "val_loss", loss)
    return loss
  
  # Define prediction method (because the default just runs forward(), and it
  # doesn't have Sigmoid activation)
  def predict_step(self, batch, batch_idx):
    
    # Run the forward propagation, apply Sigmoid activation
    return torch.nn.Sigmoid(self.network(x.view(x.size(0), -1)))
    
  # Define optimization algorithm, LR scheduler
  def configure_optimizers(self):
    
    # Optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
    
    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
      optimizer, 
      base_lr = learning_rate, max_lr = (learning_rate * 10), step_size_up = 400,
      cycle_momentum = False, mode = "exp_range", gamma = 0.8, verbose = True)
    
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
  train_data, batch_size = 64, num_workers = 4, shuffle = True)
val_loader = torch.utils.data.DataLoader(
  val_data, batch_size = 64, num_workers = 4, shuffle = False)


# Create trainer
trainer = pl.Trainer(
  max_epochs = 100,
  accelerator = "gpu", precision = "16-mixed", 
  callbacks = pl.callbacks.EarlyStopping(
    monitor = "val_loss",
    min_delta = 1e-3,
    patience = 5
    )
  )


# Train model
model = TwoHiddenLayers()
trainer.fit(model, train_loader, val_loader)
    
