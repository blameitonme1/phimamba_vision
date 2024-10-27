import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import ViTModel, ViTConfig
from utils.config import Config
import time
from modules.vm_head import VMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load dataset
dataset = load_dataset("imagenet-1k", split="train[:2000]")
# load only part of the dataset
test_dataset = load_dataset("imagenet-1k", split="validation[:2000]")

# 使用 Hugging Face 的 ViTImageProcessor 进行数据预处理
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transform_dataset(example):
    inputs = image_processor(
        [img.convert("RGB") for img in example['image']],
        return_tensors='pt',
        do_resize=True,
        size={'height': 448, 'width': 448},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    example['pixel_values'] = inputs['pixel_values']
    inputs = image_processor(
        [img.convert("RGB") for img in example['image']],
        return_tensors='pt',
        do_resize=True,
        size={'height': 224, 'width': 224},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    example['pixel_values_224'] = inputs['pixel_values']
    return example

dataset = dataset.map(transform_dataset, batched=True)
test_dataset = test_dataset.map(transform_dataset, batched=True)

# 设置数据集格式
dataset.set_format(type='torch', columns=['pixel_values', 'pixel_values_224', 'label'])
test_dataset.set_format(type='torch', columns=['pixel_values', 'pixel_values_224',  'label'])

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# teacher model
teacher_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)


mlp_weights_from_teacher = []

for name, param in teacher_model.named_parameters():
    if "intermediate" in name or "output" in name:
        if "bias" not in name and "attention" not in name:
            # print(f"found mlp in layer {name}")
            mlp_weights_from_teacher.append(param)

# student model
model_config = Config.from_json("./vm.json")
student_model = VMHeadModel(model_config).to(device)

# initialize mlp component from teacher model
layer_idx = 0
for name, param in student_model.named_parameters():
    if any(
        [
            n in name
            for n in [
                "mlp",
            ]
        ]
    ):
        param.requires_grad_(False)
        if "mlp" in name and "bias" not in name:
            param.data = mlp_weights_from_teacher[layer_idx].data
            layer_idx += 1
    else:
        param.requires_grad_(True)

param1 = sum(p.numel() for p in teacher_model.parameters())
param2 = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
print(f"teacher parameters: {param1}")
print(f"student parameters: {param2}")

T = 4.0
student_model.requires_grad_(True)

optimizer = optim.AdamW(student_model.parameters(), lr=3e-5)
# lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)

criterion_kl = nn.KLDivLoss(reduction="batchmean")
criterion_ce = nn.CrossEntropyLoss()

student_model.train()
alpha = 0.7
for epoch in range(150):
    for batch in tqdm(dataloader):
        images = batch['pixel_values'].to(device)
        images_224 = batch['pixel_values_224'].to(device)
        labels = batch['label'].to(device, dtype=torch.long)

        with torch.no_grad():
            teacher_outputs = teacher_model(images_224).logits
            teacher_outputs = nn.functional.softmax(teacher_outputs / T, dim=1)

        student_outputs = student_model(images).logits
        student_outputs = nn.functional.log_softmax(student_outputs / T, dim=1)

        loss_kl = criterion_kl(student_outputs, teacher_outputs) * T * T
        loss_ce = criterion_ce(student_outputs, labels)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write(f"Epoch {epoch+1}, lr {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.4f}\n")
        sys.stdout.flush()

    # lr_scheduler.step()

student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        images = batch['pixel_values'].to(device)
        labels = batch['label'].to(device, dtype=torch.long)

        student_outputs = student_model(images).logits
        _, predicted = torch.max(student_outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(student_model.state_dict(), "model_imagenet.pth")

print("DONE")
