# Used cars kicks classification - NN implementation in PyTorch
# Data source: 
# https://www.openml.org/search?type=data&sort=runs&id=41162&status=active
# https://www.kaggle.com/competitions/DontGetKicked/overview


# Source previous script
exec(open("04_Preprocessing.py").read())


import torch
from sklearn.utils.class_weight import compute_class_weight


# Set Torch settings
torch.set_default_dtype(torch.float32)


# Apply scikit preprocessing pipeline
x_train = pipe_process.fit_transform(x_train, y_train)
x_test = pipe_process.transform(x_test)


# Compute class weight
classes = list(set(y_train))
class_weight = compute_class_weight("balanced", classes = classes, y = y_train)
class_weight = torch.tensor(class_weight[1], dtype = torch.float32)


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
 
 
# Load data
train_data = TorchDataset(x_train, y_train)
val_data = TorchDataset(x_test, y_test)


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True)


# Get training device
device = ("cuda" if torch.cuda.is_available() else "cpu")


# Define model, inheriting from nn.Module class
input_size = 88
hidden_size = 64
output_size = 64

class TwoHiddenLayers(torch.nn.Module):
  
  # Define model architecture
  def __init__(self):
    super().__init__() # Delegate function to parent class
    
    self.network = torch.nn.Sequential(
      torch.nn.Linear(in_features = input_size, out_features = hidden_size), # Hidden 1
      torch.nn.ReLU(), # Activation 1
      torch.nn.Linear(in_features = hidden_size,  out_features = hidden_size), # Hidden 2
      torch.nn.ReLU(), # Activation 2
      torch.nn.Linear(in_features = output_size, out_features = 1) # Output layer
      # No Sigmoid activation here because the loss function has it built-in
    )
  
  # Define forward propagation
  def forward(self, x):
    output = self.network(x)
    return output
    

# Create model
model_nn = TwoHiddenLayers()


# Define hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 10


# Define loss function with class weights, optimizer
# This loss function applies built-in sigmoid activation
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = class_weight, reduction = "mean")
optimizer = torch.optim.Adam(model_nn.parameters(), lr = learning_rate)


# Define training loop
def train_loop(train_loader, model_nn, loss_fn, optimizer):
  
  size = len(train_loader.dataset) # Get n. of training obs.
  
  for batch, (x, y) in enumerate(train_loader): # For each training batch
    
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
