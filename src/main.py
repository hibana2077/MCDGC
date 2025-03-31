import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set Seed
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple CNN model supports configurable input channels & image size.
class DropoutCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, in_channels=1, image_size=28):
        super(DropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        # Calculate the size after three pooling layers
        fc_input_dim = 128 * ((image_size // 8) ** 2)
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # save for grad-cam
        self.activations = []
        self.gradients = []
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Save final conv layer output for Grad-CAM
        x = F.relu(self.conv3(x))
        self.activations.append(x)
        
        # Register hook to capture gradients
        x.register_hook(lambda grad: self.gradients.append(grad))
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients[-1]
    
    def get_activations(self):
        return self.activations[-1]

# Load dataset (MNIST or CIFAR-10)
def load_data(dataset="mnist"):
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

# Train model
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    print("Training completed!")
    return model

# Compute Grad-CAM
def compute_gradcam(model, input_image, target_class=None):
    model.eval()
    
    output = model(input_image)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    output.backward(gradient=one_hot, retain_graph=True)
    
    gradients = model.get_activations_gradient()
    activations = model.get_activations()
    
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-10)
    
    cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
    
    return cam.detach().cpu().numpy()[0, 0]

# Monte Carlo Dropout Grad-CAM
def mc_dropout_gradcam(model, input_image, num_samples=30, target_class=None):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    cam_samples = []
    for _ in range(num_samples):
        cam = compute_gradcam(model, input_image, target_class)
        cam_samples.append(cam)
    
    cam_samples = np.array(cam_samples)
    mean_cam = np.mean(cam_samples, axis=0)
    std_cam = np.std(cam_samples, axis=0)
    
    return mean_cam, std_cam, cam_samples

# Visualize results (handles both grayscale and color images)
def visualize_results(image, mean_cam, std_cam, samples=None, num_samples_to_show=5):
    if image.shape[0] == 3:
        # Convert from (3, H, W) to (H, W, 3)
        orig_image = np.transpose(image.cpu().squeeze().numpy(), (1, 2, 0))
        # Bring to [0,1] range for visualization
        orig_image = (orig_image - orig_image.min()) / (orig_image.max() - orig_image.min() + 1e-10)
    else:
        orig_image = image.cpu().squeeze().numpy()

    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    if image.shape[0] == 3:
        plt.imshow(orig_image)
    else:
        plt.imshow(orig_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Mean Grad-CAM")
    if image.shape[0] == 3:
        plt.imshow(orig_image)
    else:
        plt.imshow(orig_image, cmap='gray')
    plt.imshow(mean_cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Uncertainty (Std)")
    plt.imshow(std_cam, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Mean CAM + Uncertainty")
    alpha = std_cam / np.max(std_cam)
    if image.shape[0] == 3:
        plt.imshow(orig_image)
    else:
        plt.imshow(orig_image, cmap='gray')
    plt.imshow(mean_cam, cmap='jet', alpha=0.7 * (1 - alpha))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mc_gradcam_result.png')
    plt.show()
    
    if samples is not None and num_samples_to_show > 0:
        plt.figure(figsize=(15, 3))
        for i in range(min(num_samples_to_show, len(samples))):
            plt.subplot(1, num_samples_to_show, i+1)
            plt.title(f"Sample {i+1}")
            if image.shape[0] == 3:
                plt.imshow(orig_image)
            else:
                plt.imshow(orig_image, cmap='gray')
            plt.imshow(samples[i], cmap='jet', alpha=0.5)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('mc_gradcam_samples.png')
        plt.show()

# Main function with option for dataset choice
def main():
    # Choose dataset: "mnist" or "cifar10"
    dataset_choice = "cifar10"  # change to "mnist" for MNIST
    
    # Load data
    train_loader, test_loader = load_data(dataset=dataset_choice)
    
    # Set parameters based on dataset
    if dataset_choice == "cifar10":
        in_channels = 3
        image_size = 32
    else:
        in_channels = 1
        image_size = 28
    
    # Create model
    model = DropoutCNN(dropout_rate=0.5, in_channels=in_channels, image_size=image_size).to(device)
    
    # Train model
    model = train_model(model, train_loader, epochs=10)
    
    torch.save(model.state_dict(), 'dropout_cnn.pth')
    
    # Get one sample from test set
    idx = random.randint(0, len(test_loader.dataset) - 1)
    test_image, test_label = test_loader.dataset[idx]
    test_image = test_image.unsqueeze(0).to(device)
    
    # Execute Monte Carlo Dropout Grad-CAM
    mean_cam, std_cam, cam_samples = mc_dropout_gradcam(model, test_image, num_samples=60)
    
    # Visualize results
    visualize_results(test_image[0], mean_cam, std_cam, cam_samples)
    
    print(f"True Label: {test_label}")
    
    model.eval()
    output = model(test_image)
    pred = output.argmax(dim=1).item()
    print(f"Prediction Label: {pred}")

if __name__ == "__main__":
    main()