import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set Seed
torch.manual_seed(42)
np.random.seed(42)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# simple cnn model
class DropoutCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # save for grad-cam
        self.activations = []
        self.gradients = []
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 保存最後一個卷積層的輸出，用於 Grad-CAM
        x = F.relu(self.conv3(x))
        self.activations.append(x)
        
        # 註冊鉤子函數，獲取梯度
        h = x.register_hook(lambda grad: self.gradients.append(grad))
        
        x = self.pool(x)
        
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    # 設定 Grad-CAM 目標層的梯度和輸出
    def get_activations_gradient(self):
        return self.gradients[-1]
    
    def get_activations(self):
        return self.activations[-1]

# 載入 MNIST 數據集
def load_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

# 訓練模型
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

# 計算 Grad-CAM
def compute_gradcam(model, input_image, target_class=None):
    # 設置模型為評估模式，但保持 dropout 啟用
    model.eval()
    
    # 前向傳播
    output = model(input_image)
    
    # 如果沒有指定目標類別，使用預測的類別
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # 反向傳播
    model.zero_grad()
    
    # 只對目標類別進行反向傳播
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    
    # 反向傳播
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 獲取梯度和特徵圖
    gradients = model.get_activations_gradient()
    activations = model.get_activations()
    
    # 對梯度進行全局平均池化
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    # 生成 Grad-CAM
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)  # 應用 ReLU
    
    # 正規化 CAM
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-10)
    
    # 調整大小與輸入圖像一致
    cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
    
    return cam.detach().cpu().numpy()[0, 0]

# Monte Carlo Dropout Grad-CAM
def mc_dropout_gradcam(model, input_image, num_samples=30, target_class=None):
    # 啟用 dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    # 執行多次 Grad-CAM
    cam_samples = []
    for _ in range(num_samples):
        cam = compute_gradcam(model, input_image, target_class)
        cam_samples.append(cam)
    
    # 計算平均 CAM 和不確定性 (標準差)
    cam_samples = np.array(cam_samples)
    mean_cam = np.mean(cam_samples, axis=0)
    std_cam = np.std(cam_samples, axis=0)
    
    return mean_cam, std_cam, cam_samples

# 可視化結果
def visualize_results(image, mean_cam, std_cam, samples=None, num_samples_to_show=5):
    plt.figure(figsize=(15, 8))
    
    # 顯示原始圖像
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # 顯示平均 Grad-CAM
    plt.subplot(1, 4, 2)
    plt.title("Mean Grad-CAM")
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow(mean_cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # 顯示不確定性 (標準差)
    plt.subplot(1, 4, 3)
    plt.title("Uncertainty (Std)")
    plt.imshow(std_cam, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 疊加平均 Grad-CAM 和不確定性
    plt.subplot(1, 4, 4)
    plt.title("Mean CAM + Uncertainty")
    # 將不確定性轉換為顏色透明度
    alpha = std_cam / np.max(std_cam)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow(mean_cam, cmap='jet', alpha=0.7 * (1 - alpha))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mc_gradcam_result.png')
    plt.show()
    
    # 顯示樣本 CAM (可選)
    if samples is not None and num_samples_to_show > 0:
        plt.figure(figsize=(15, 3))
        for i in range(min(num_samples_to_show, len(samples))):
            plt.subplot(1, num_samples_to_show, i+1)
            plt.title(f"Sample {i+1}")
            plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
            plt.imshow(samples[i], cmap='jet', alpha=0.5)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('mc_gradcam_samples.png')
        plt.show()

# 主函數
def main():
    # 載入數據
    train_loader, test_loader = load_data()
    
    # 創建模型
    model = DropoutCNN(dropout_rate=0.5).to(device)
    
    # 訓練模型
    model = train_model(model, train_loader, epochs=3)
    
    # 儲存模型
    torch.save(model.state_dict(), 'dropout_cnn.pth')
    
    # 加載模型 (可選)
    # model = DropoutCNN(dropout_rate=0.5).to(device)
    # model.load_state_dict(torch.load('dropout_cnn.pth'))
    
    # 從測試集獲取一個範例
    for test_images, test_labels in test_loader:
        test_image = test_images.to(device)
        test_label = test_labels.item()
        break
    
    # 執行 Monte Carlo Dropout Grad-CAM
    mean_cam, std_cam, cam_samples = mc_dropout_gradcam(model, test_image, num_samples=30)
    
    # 可視化結果
    visualize_results(test_image, mean_cam, std_cam, cam_samples)
    
    print(f"真實標籤: {test_label}")
    
    # 預測標籤
    model.eval()
    with torch.no_grad():
        output = model(test_image)
        pred = output.argmax(dim=1).item()
    print(f"預測標籤: {pred}")

if __name__ == "__main__":
    main()