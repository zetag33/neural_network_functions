import torch
from torch import nn

class MyLinearModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.linear_layer(x)






def training_loop(X_train, y_train, X_test, y_test, epochs, lr):
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = MyLinearModel()
  model = model.to(device)
  params = mymodel.parameters()
  loss_function = nn.L1Loss()
  optimizer = torch.optim.SGD(params, lr)
  X_train = X_train.to(device)
  y_train = y_train.to(device)
  X_test = X_test.to(device)
  y_test = y_test.to(device)
  
  for epoch in range(epochs):
    model.train()
    # Forward pass
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    optimizer.step()
    # test
    mymodel.eval()
    with torch.inference_mode():
      y_pred = mymodel.forward(X_test)
      test_loss = loss_function(y_pred,y_test)
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
