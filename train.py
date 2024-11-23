import os
import torch
from datetime import datetime
from mnist_cnn import SimpleCNN, train_epoch

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    print("Starting training...")
    model = SimpleCNN()
    train_epoch()
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'models/model_{timestamp}.pth')
    print(f"Model saved as model_{timestamp}.pth")

if __name__ == "__main__":
    main() 