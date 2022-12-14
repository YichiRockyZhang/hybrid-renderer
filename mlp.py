from torch import nn

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(198*768, 224*224),
      nn.ReLU(),
      nn.Linear(224*224, 224*224)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)