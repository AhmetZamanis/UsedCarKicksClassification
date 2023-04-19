# Used cars kicks classification - Modeling, Neural Network with PyTorch
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import torch
import optuna
from sklearn.utils.class_weight import compute_class_weight


# Set Torch settings
torch.set_default_dtype(torch.float32)


# Apply scikit preprocessing pipeline
x_train = pipe_process.fit_transform(x_train, y_train)
x_test = pipe_process.transform(x_test)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)


# Define Dataset class: Takes in preprocessed features & targets
class TorchDataset(torch.utils.data.Dataset):
  
  # Store preprocessed features & targets
  def __init__(self, x_train, y_train):
    self.x = torch.tensor(x_train, dtype = torch.float32) # Store features
    self.y = torch.tensor(y_train.values, dtype = torch.int32) # Store targets
  
  # Return data length  
  def __len__(self):
    return len(self.x) 
  
  # Return a pair of features & target
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
 
 
# Load data
train_data = TorchDataset(x_train, y_train)
val_data = TorchDataset(x_test, y_test)


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True)


# One iteration of dataloader
train_x, train_y = next(iter(train_loader))
train_x.size()
train_y.size()  


# Get training device
device = ("cuda" if torch.cuda.is_available() else "cpu")


# Define model, inheriting from nn.Module class
input_size = 88
hidden_size1 = 64
hidden_size2 = 64
output_size = 64

class TwoHiddenLayers(torch.nn.Module):
  
  # Define model architecture
  def __init__(self):
    super().__init__() # Delegate function to parent class
    
    self.network = torch.nn.Sequential(
      torch.nn.Linear(in_features = 88, out_features = hidden_size1), # Input layer
      torch.nn.ReLU(), # Activation 1
      torch.nn.Linear(in_features = hidden_size1, out_features = hidden_size2), # Hidden 1
      torch.nn.ReLU(), # Activation 2
      torch.nn.Linear(in_features = hidden_size2,  out_features = output_size), # Hidden 2
      torch.nn.Linear(in_features = output_size, out_features = 1) # Output layer
      # torch.nn.Sigmoid() # Sigmoid activation for prob. output  
    )
  
  # Define forward propagation
  def forward(self, x):
    prob = self.network(x)
    return prob
    

# Create model
model_nn = TwoHiddenLayers()
model_nn.network
print(torch.nn.Linear(in_features = 88, out_features = hidden_size1).weight.dtype)


# Define hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 10


# Define loss function with class weights, optimizer
# loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weight, dtype = torch.float32), reduction = "mean")
loss_fn = torch.nn.BCEWithLogitsLoss(
  pos_weight = torch.tensor(class_weight[1], dtype = torch.float32), 
  reduction = "mean")
optimizer = torch.optim.Adam(model_nn.parameters(), lr = learning_rate)


# Define training loop
def train_loop(train_loader, model_nn, loss_fn, optimizer):
  
  size = len(train_loader.dataset) # Get n. of training obs.
  
  for batch, (x, y) in enumerate(train_loader): # For each training batch
    
    y = y.unsqueeze(1).float()
    
    # Perform training, calculate loss
    pred = model_nn(x)
    loss = loss_fn(pred, y)
    
    # Perform backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Report batch training loss, n. of training obs seen
    if batch % 100 == 0:
      loss, obs_seen = loss.item(), (batch + 1) * len(x)
      print(f"Train loss: {loss:>5f} [{obs_seen:>5d}/{size:>5d}]")


# Define validation loop
def val_loop(val_loader, model_nn, loss_fn):
  
  n_batches = len(val_loader) # Get number of val batches
  test_loss = 0
  
  with torch.no_grad(): # Disable gradient calculation for testing loop
    for x, y in val_loader: # For each validation batch
      
      y = y.unsqueeze(1).float()
      
      pred = model_nn(x) # Make prediction
      test_loss += loss_fn(pred, y).item() # Calculate loss
  
  test_loss = test_loss / n_batches # Calculate avg. batch loss
  print(f"Avg. test loss: {test_loss:>6f}")


# Perform training with validation set
for epoch in range(epochs):
  print(f"Epoch: {epoch + 1}\n-----------")
  train_loop(train_loader, model_nn, loss_fn, optimizer)
  val_loop(val_loader, model_nn, loss_fn)
print("Done")
