import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import math
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from models import *  # 假设这里包含了VMHeadModel定义
from utils.config import Config  # 假设这里包含了从json加载配置的功能
from modules.vm_head import VMHeadModel
from transformers import ViTModel, ViTConfig

# 定义模型
# class StudentModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#             nn.Linear(186624, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         return self.model(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
student_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
# 加载模型配置
student_config.num_hidden_layers = 1
student_config.num_labels = 10
model_config = Config.from_json("./vm.json")
# student_model = VMHeadModel(model_config).to(device)
# student_model = ViTModel(student_config).to(device)
student_model = ViTForImageClassification(student_config).to(device)
# student_model = StudentModel().to(device)
student_model.requires_grad_(True)
print(f"parameter count: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}")
# student_model = StudentModel().to(device)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

# student_model.apply(init_weights)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像转换为三通道图像
    # transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for name, param in student_model.named_parameters():
    if param.requires_grad:
        print(name)
    else:
        raise ValueError("No grad")

# 定义优化器和损失函数
optimizer = optim.AdamW(student_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练模型
student_model.train()
for epoch in range(1):
    for (images, labels) in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        student_outputs = student_model(images).logits
        # student_outputs = student_model(images)
        # print(student_outputs)
        loss = criterion(student_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(student_model.vm_head.weight.grad)
        # if random.random() < 0.01:
        sys.stdout.write(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
        sys.stdout.flush()

student_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        student_outputs = student_model(images).logits
        _, predicted = torch.max(student_outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

print("DONE")
