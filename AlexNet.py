import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,WeightedRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# 定义AlexNet模型

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            #nn.Softmax(dim=1),  # 添加 Softmax 层
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 设置随机种子
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据加载和预处理
train_dataset = ImageFolder("data_sex/train", transform=transform)
test_dataset = ImageFolder("data_sex/test", transform=transform)


targets = torch.tensor(train_dataset.targets)
# 统计每个类别的样本数量
class_counts = torch.bincount(targets)

# 计算每个类别的权重
class_weights = 1.0 / class_counts.float()

# 根据类别权重创建采样器
weights = class_weights[targets]
sampler = WeightedRandomSampler(weights, len(weights))


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 创建AlexNet模型实例
model = AlexNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 在训练集上计算准确率
    correct = 0
    total = 0

    # 在训练集上计算准确率和损失函数值
    model.eval()  # 设置模型为评估模式
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += criterion(outputs, labels).item()

    train_accuracy = 100 * train_correct / train_total
    train_loss = train_loss / len(train_loader)

    print(f"Train Accuracy: {train_accuracy:.2f}%")
    #print(f"Train Loss: {train_loss:.4f}")

    # 在验证集上计算准确率和损失函数值
    model.eval()  # 设置模型为评估模式
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

    test_accuracy = 100 * test_correct / test_total
    test_loss = test_loss / len(test_loader)

    print(f"Test Accuracy: {test_accuracy:.2f}%")

print(f"Test Loss: {test_loss:.4f}")

# 在验证集上计算预测结果和真实标签
model.eval()  # 设置模型为评估模式
test_predicted = []
test_true = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        test_predicted.extend(predicted.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

# 计算混淆矩阵
confusion_mat = confusion_matrix(test_true, test_predicted)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
class_names = test_dataset.classes
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()