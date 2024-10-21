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
from models import *  # 假设这里包含了VMHeadModel定义
from utils.config import Config  # 假设这里包含了从json加载配置的功能
from modules.vm_head import VMHeadModel
device = "cuda" if torch.cuda.is_available() else "cpu"

teacher_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

teacher_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
teacher_config.num_labels = 10
teacher_model.classifier = nn.Linear(teacher_config.hidden_size, teacher_config.num_labels).to(device)

model_config = Config.from_json("./vm.json")
student_model = VMHeadModel(model_config).to(device)

mlp_weights_from_teacher = []

for name, param in teacher_model.named_parameters():
    if "intermediate" in name or "output" in name:
        if "bias" not in name and "attention" not in name:
            # print(f"found mlp in layer {name}")
            mlp_weights_from_teacher.append(param)

student_model.requires_grad_(True)
layer_idx = 0
for name, param in student_model.named_parameters():
    if any(
        [
            n in name
            for n in [
                "mlp",
                # "input_layernorm",
                # "embedding",
                # "final_layernorm",
                # "vm_head",
            ]
        ]
    ):
        param.requires_grad_(False)
        if "mlp" in name and "bias" not in name:
            # print(name)
            param.data = mlp_weights_from_teacher[layer_idx].data
            layer_idx += 1
    else:
        param.requires_grad_(True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=5e-4)
# criterion_kl = nn.KLDivLoss(reduction="batchmean")
criterion_ce = nn.CrossEntropyLoss()
student_model.train()

# check trainable parameters

# for name, param in student_model.named_parameters():
#     if param.requires_grad:
#         print(name)

for epoch in range(1):
    for (images, labels) in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)

        # with torch.no_grad():
        #     teacher_outputs = teacher_model(images).last_hidden_state # logits -> log probabilities
        #     teacher_outputs = nn.functional.softmax(teacher_outputs, dim=1)

        student_outputs = student_model(images).logits
        # student_outputs = nn.functional.log_softmax(student_outputs, dim=1)  # log probabilities
        # student_outputs = torch.argmax(student_outputs, dim=1)
        # change dtype
        # print(student_outputs.shape, labels.shape)
        # loss_kl = criterion_kl(student_outputs, teacher_outputs)
        loss_ce = criterion_ce(student_outputs, labels.to(device))
        loss = loss_ce
        sys.stdout.write(f"{loss:.4f}\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("DONE")