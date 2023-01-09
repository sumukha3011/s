class netC(nn.Module):
    def __init__(self):
        super(netC, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=4,out_channels=10,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        self.conv3=nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu3=nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=20,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.relu4=nn.ReLU()
        self.fc1=nn.Linear(in_features=24*7*7,out_features=300)
        self.fc2=nn.Linear(300,100)
        self.fc3=nn.Linear(100,10)
        self.out_=nn.LogSoftmax(dim=1)
    
