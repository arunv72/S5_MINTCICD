import sys
import os
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_cnn import SimpleCNN

@pytest.fixture
def model():
    return SimpleCNN()

@pytest.fixture
def sample_input():
    return torch.randn(1, 1, 28, 28)

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 26000, f"Model has {total_params} parameters, should be less than 26000"

def test_input_shape(model, sample_input):
    try:
        output = model(sample_input)
        assert True, "Model accepts 28x28 input"
    except:
        assert False, "Model failed to process 28x28 input"

def test_output_shape(model, sample_input):
    output = model(sample_input)
    assert output.shape[1] == 10, f"Model output should have 10 classes, got {output.shape[1]}"

def test_model_accuracy():
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Setup model
    model = SimpleCNN()
    
    # Load trained model if exists
    model_files = [f for f in os.listdir('./models') if f.endswith('.pth')]
    if not model_files:
        pytest.skip("No trained model found to test accuracy")
    
    latest_model = max(model_files)
    model.load_state_dict(torch.load(f'./models/{latest_model}'))
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%" 