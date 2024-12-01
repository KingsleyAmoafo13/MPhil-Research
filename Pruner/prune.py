import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser to specify the backbone model
parser = argparse.ArgumentParser(description="Prune and Fine-tune Pre-trained Model")
parser.add_argument("--backbone", type=str, required=True, choices=['vgg16', 'resnet50', 'resnet101'],
                    help="Specify the backbone model to prune. Choices: vgg16, resnet50, resnet101")
args = parser.parse_args()

# Hyperparameters
prune_percent = 0.3  # Prune 30% of weights
l2_lambda = 1e-4     # L2 regularization strength
fine_tune_epochs = 5 # Fine-tuning epochs
batch_size = 32      # Batch size

# Path to save pruned and fine-tuned models
save_path = "C:/Users/Aking/Desktop/MPHIL_RESEARCH_3/best_models"

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load a dataset (e.g., CIFAR-10)
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load Pretrained Models (VGG16, ResNet50, ResNet101)
models_dict = {
    'vgg16': vgg.vgg16(pretrained=True),
    'resnet50': resnet.resnet50(pretrained=True),
    'resnet101': resnet.resnet101(pretrained=True)
}

# Function to count the total and non-zero parameters in a model
def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum(p.nonzero().size(0) for p in model.parameters())
    return total_params, non_zero_params

# Magnitude-based pruning function
def prune_model(model, prune_percent):
    total_params_before, _ = count_params(model)
    print(f"Total parameters before pruning: {total_params_before}")
    
    # Flatten all weights and sort by magnitude
    all_weights = torch.cat([param.view(-1) for param in model.parameters()])
    sorted_weights, _ = torch.sort(torch.abs(all_weights))
    
    # Determine the threshold below which weights will be pruned
    params_to_prune = int(all_weights.numel() * prune_percent)
    threshold = sorted_weights[params_to_prune]
    
    # Apply pruning (set weights below the threshold to zero)
    for param in model.parameters():
        param_mask = torch.abs(param) >= threshold
        param.data *= param_mask.float()
    
    _, non_zero_params_after = count_params(model)
    print(f"Non-zero parameters after pruning: {non_zero_params_after}")

# Fine-tuning function with L2 regularization
def fine_tune_model(model, train_loader, epochs, l2_lambda):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(train_loader))
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Function to prune and fine-tune a model, and save it
def prune_and_finetune(model_name):
    print(f"\nPruning and fine-tuning {model_name.upper()}...\n")
    
    # Get the model
    model = models_dict[model_name]
    model.to(device)
    
    # Prune the model
    prune_model(model, prune_percent)
    
    # Fine-tune the model
    fine_tune_model(model, train_loader, fine_tune_epochs, l2_lambda)
    
    # Save the pruned and fine-tuned model
    model_save_path = os.path.join(save_path, f"{model_name}_pruned_finetuned.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Prune and fine-tune the specified backbone model
prune_and_finetune(args.backbone)
