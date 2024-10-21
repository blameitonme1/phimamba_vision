import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from utils.config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载教师模型
teacher_model = PhiForCausalLM.from_pretrained(
    "microsoft/phi-1_5", attn_implementation="eager"
).to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

# 加载学生模型配置
model_config = Config.from_json("assets/sample_config.json")
student_model = LMHeadModel(model_config).to(device)

# 定义视觉数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# 定义优化器
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 训练阶段
student_model.train()
for epoch in range(1):  # 只运行一个epoch作为示例
    for idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # 使用教师模型生成输入
        input_ids = (
            tokenizer(images.view(-1, 32*32*3).tolist(), return_tensors="pt", truncation=True)
            .to(device)
            .input_ids
        )

        _, seq_len = input_ids.size()

        teacher_outputs = teacher_model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

        # 清零梯度
        optimizer.zero_grad()

        for layer_idx, student_layer in enumerate(student_model.backbone.layers):
            student_input = teacher_outputs.hidden_states[layer_idx]

            # 前向传播
            student_output = student_layer(
                hidden_states=student_input,
                run_mlp_component=False,
                return_mixer_matrix=True,
            )
            transfer_matrix = student_output["transfer_matrix"]
            attn_matrix = teacher_outputs.attentions[layer_idx]

            assert transfer_matrix.size() == attn_matrix.size()

            loss = torch.linalg.matrix_norm(
                transfer_matrix - attn_matrix, ord="fro"
            ).mean()

            loss.backward()
            print(f"Epoch {epoch}, Iter {idx}, Layer {layer_idx}, Loss: {loss.item()}")

        # 更新参数
        optimizer.step()

print("DONE")