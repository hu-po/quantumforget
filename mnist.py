import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic
import copy


EPOCHS = 3
BATCH_SIZE = 64
LR=0.001
QLEVELS = [
    # torch.float16,
    torch.qint8,
]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define data transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def model_to_str(model):
    components = []

    state_dict = model.state_dict()
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            components.append(",".join(map(str, value.cpu().float().numpy().flatten())))
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, torch.Tensor):
                    components.append(",".join(map(str, item.cpu().float().numpy().flatten())))

    return ";".join(components)

def str_to_model(model_str, template_model):
    model = copy.deepcopy(template_model)
    components = model_str.split(";")

    idx = 0
    new_state_dict = model.state_dict()

    for key, value in new_state_dict.items():
        if isinstance(value, torch.Tensor):
            data = components[idx].split(",")
            new_state_dict[key] = torch.tensor([float(v) for v in data], dtype=value.dtype).view_as(value)
            idx += 1
        elif isinstance(value, tuple):
            items = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    data = components[idx].split(",")
                    items.append(torch.tensor([float(v) for v in data], dtype=item.dtype).view_as(item))
                    idx += 1
            new_state_dict[key] = tuple(items)

    model.load_state_dict(new_state_dict)
    return model

def compare_models(model_a, model_b):
    model_a_sd = model_a.state_dict()
    model_b_sd = model_b.state_dict()

    all_equal = True
    for key in model_a_sd:
        if not torch.equal(model_a_sd[key], model_b_sd[key]):
            all_equal = False
            print(f"Warning: Mismatch detected for {key}")
            print("Model A Value:", model_a_sd[key])
            print("Model B Value:", model_b_sd[key])
            print("="*50)
    return all_equal

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter('./logs')

model_str_trajs = {
    _qlevel : [] for _qlevel in QLEVELS
}

# Train the model
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Log train loss every batch
        writer.add_scalar('Train loss', loss.item(), i + epoch * len(train_loader))

    # Model can be saved out as a string
    model_str = model_to_str(model)
    print(model_str)
    assert compare_models(str_to_model(model_str, SimpleNN()), model)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    writer.add_scalar('test_acc/raw', test_accuracy, epoch)

    # Quantize the model
    for qlevel in QLEVELS:
        quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=qlevel)

        # Log the model string trajectory
        model_str_trajs[qlevel].append(model_to_str(quantized_model))

        # Test the quantized model
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = quantized_model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        writer.add_scalar(f"test_acc/{qlevel}", test_accuracy, epoch)

    print('Epoch: {0:2d} Test accuracy: {1:.2f}%'.format(epoch, test_accuracy))

# Log the model string trajectory
for qlevel in QLEVELS:
    for i, model_str in enumerate(model_str_trajs[qlevel]):
        print(f"Epoch {i} weights: {model_str}")

writer.close()