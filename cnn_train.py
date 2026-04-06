import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
from tqdm import tqdm

# ==================== 配置参数 ====================
# 【重要修改】指向包含 train/val/test 子文件夹的根目录
DATA_ROOT = r"D:\rice-disease-det\datasets"

MODEL_NAME = "resnet50"  # 可选：resnet18, resnet50, mobilenet_v2, efficientnet_b0
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强 (训练集)
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据标准化 (验证集/测试集/推理)
VAL_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 控制流程
RUN_TRAIN = True
RUN_VAL = True
RUN_TEST = True
SAVE_MODEL = True
INFERENCE_IMAGE = r"D:\rice-disease-det\static\images\daowenbing.jpg"


# =================================================

def get_model(model_name, num_classes):
    """加载预训练模型并修改最后一层全连接层"""
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ResNet: {model_name}")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def main():
    print(f"当前设备: {DEVICE}")
    print(f"数据集根目录: {DATA_ROOT}")

    # 1. 构建数据集路径
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')
    test_dir = os.path.join(DATA_ROOT, 'test')

    # 检查目录是否存在
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ 错误：未找到 train 或 val 文件夹！请检查 DATA_ROOT 路径。")
        return

    # 2. 加载数据 (ImageFolder 会自动根据子文件夹名分配标签)
    # 注意：classes 列表是按字母顺序排序的
    train_dataset = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(val_dir, transform=VAL_TEST_TRANSFORMS)
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=VAL_TEST_TRANSFORMS)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\n📂 检测到 {num_classes} 个类别:")
    for i, name in enumerate(class_names):
        print(f"   [{i}] {name}")

    print(f"\n📊 数据量统计:")
    print(f"   训练集: {len(train_dataset)} 张")
    print(f"   验证集: {len(val_dataset)} 张")
    if test_dataset:
        print(f"   测试集: {len(test_dataset)} 张")

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2,pin_memory=True) if test_dataset else None

    # 4. 初始化模型、损失函数、优化器
    model = get_model(MODEL_NAME, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率衰减策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    model_path = rf"D:\rice-disease-det\models\cnn\best_{MODEL_NAME}.pt"

    # 5. 训练循环
    if RUN_TRAIN:
        print("\n🚀 开始训练...")
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            print(f"Epoch {epoch + 1:03d}/{EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                if SAVE_MODEL:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'class_names': class_names,
                        'acc': best_acc
                    }, model_path)
                    print(f"   💾 已保存最佳模型 ({model_path}) - Acc: {best_acc:.4f}")

        print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.4f}")

    # 6. 测试集评估
    if RUN_TEST and test_loader:
        print("\n🧪 开始在测试集上评估...")
        # 加载最佳权重进行测试
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   已加载权重: {model_path}")
        else:
            print("   ⚠️ 未找到保存的最佳模型，使用当前训练结束的权重")

        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
        print(f"🏆 测试集结果 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    #
    # # 7. 单张图片推理
    # if INFERENCE_IMAGE and os.path.exists(INFERENCE_IMAGE):
    #     print(f"\n🔍 单图推理示例: {os.path.basename(INFERENCE_IMAGE)}")
    #     model.eval()
    #
    #     # 预处理图片
    #     image = Image.open(INFERENCE_IMAGE).convert('RGB')
    #     image_tensor = VAL_TEST_TRANSFORMS(image).unsqueeze(0).to(DEVICE)
    #
    #     with torch.no_grad():
    #         outputs = model(image_tensor)
    #         probs = torch.softmax(outputs, dim=1)
    #         confidence, pred_idx = torch.max(probs, 1)
    #
    #     pred_class_name = class_names[pred_idx.item()]
    #     print(f"   👉 预测类别: {pred_class_name}")
    #     print(f"   👉 置信度: {confidence.item():.4f}")
    #
    #     # 打印前3个可能的类别
    #     top_k = 3
    #     top_probs, top_indices = torch.topk(probs, k=min(top_k, num_classes))
    #     print(f"   📊 Top {top_k} 可能性:")
    #     for i in range(min(top_k, num_classes)):
    #         idx = top_indices[0][i].item()
    #         conf = top_probs[0][i].item()
    #         print(f"      {class_names[idx]}: {conf:.4f}")


if __name__ == "__main__":
    main()