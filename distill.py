import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from transformers import ViTForImageClassification
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig
from utils.config import Config
import time
from modules.vm_head import VMHeadModel
device = "cuda" if torch.cuda.is_available() else "cpu"


# teacher_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
# teacher_config.num_labels = 10
# teacher_config.num_hidden_layers = 1

teacher_model = ViTForImageClassification.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10').to(device)
# teacher_model = ViTForImageClassification(teacher_config).to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

model_config = Config.from_json("./vm.json")
student_model = VMHeadModel(model_config).to(device)
mlp_weights_from_teacher = []

# for name, param in teacher_model.named_parameters():
#     if "intermediate" in name or "output" in name:
#         if "bias" not in name and "attention" not in name:
#             # print(f"found mlp in layer {name}")
#             mlp_weights_from_teacher.append(param)
# show teacher parameter

param1 = 0
for n, p in teacher_model.named_parameters():
    param1 += p.numel()
param2 = 0
for n, p in student_model.named_parameters():
    if p.requires_grad:
        param2 += p.numel()

print(f"teacher parameters: {param1}")
print(f"student parameters: {param2}")
T = 3.0
student_model.requires_grad_(True)
# layer_idx = 0
# for name, param in student_model.named_parameters():
#     if any(
#         [
#             n in name
#             for n in [
#                 "mlp",
#                 # "input_layernorm",
#                 # "embedding",
#                 # "final_layernorm",
#                 # "vm_head",
#             ]
#         ]
#     ):
#         param.requires_grad_(False)
#         if "mlp" in name and "bias" not in name:
#             # print(name)
#             param.data = mlp_weights_from_teacher[layer_idx].data
#             layer_idx += 1
#     else:
#         param.requires_grad_(True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# cut dataset
# dataset = torch.utils.data.Subset(dataset, list(range(0, 1000)))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
optimizer = optim.AdamW(student_model.parameters(), lr=1e-5)
# print out the loss
criterion_kl = nn.KLDivLoss(reduction="batchmean")
criterion_ce = nn.CrossEntropyLoss()
student_model.train()

# check trainable parameter

for epoch in range(20):
    for (images, labels) in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        with torch.no_grad():
            teacher_outputs = teacher_model(images).logits
            teacher_outputs = nn.functional.softmax(teacher_outputs, dim=1)  # 应用温度

        student_outputs = student_model(images).logits
        student_outputs = nn.functional.log_softmax(student_outputs, dim=1)  # 应用温度并转换为对数概率

        # 计算KL散度损失
        loss_kl = criterion_kl(student_outputs, teacher_outputs) # 损失乘以温度的平方

        # 可选：加入交叉熵损失以保持学生模型的准确性
        # loss_ce = criterion_ce(student_outputs, labels)
        # loss = loss_kl + loss_ce

        loss = loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # set description
        sys.stdout.write(f"Epoch {epoch+1}, lr {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.4f}\n")
        sys.stdout.flush()

    # lr_scheduler.step()


# test the model
student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        student_outputs = student_model(images).logits
        _, predicted = torch.max(student_outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

print("DONE")