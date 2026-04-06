import cv2
import numpy as np
import base64
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ================= 配置数据 =================
CLASS_NAMES = [
    'Brown_Spot', 'Healthy', 'Leaf_Blast', 'Neck_Blast'
]

# RGB 颜色格式（前端列表用，后端不再需要）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
]

DISEASE_ADVICE = {
    'Brown_Spot': '【防治建议】\n1. 增施钾肥，提高抗性\n2. 清除病残体\n3. 发病初期使用丙环唑、苯醚甲环唑等',
    'Healthy': '【状态】\n植株生长健康，无需防治。继续保持良好田间管理。',
    'Leaf_Blast': '【防治建议】\n1. 选用抗病品种\n2. 合理施肥，避免过量氮肥\n3. 发病初期喷洒三环唑、稻瘟灵等\n4. 加强水浆管理',
    'Neck_Blast': '【防治建议】\n1. 选用抗病品种\n2. 破口期和齐穗期各喷药一次\n3. 药剂可选用三环唑、稻瘟灵',
}

CLASS_NAME_MAP = {
    'Brown_Spot': '褐斑病', 'Healthy': '健康叶片',
    'Leaf_Blast': '叶瘟', 'Neck_Blast': '穗颈瘟'
}

def map_class_name(english_name):
    return CLASS_NAME_MAP.get(english_name, english_name)

# ================= 编码工具函数 =================
def image_to_base64(image_np):
    """将 numpy 数组图片转换为 Base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

# ================= 模型预测逻辑 =================
_model_cache = {}

def load_model(model_path):
    if model_path in _model_cache:
        return _model_cache[model_path]
    print(f"[INFO] 加载 YOLO 模型：{model_path}")
    model = YOLO(model_path)
    _model_cache[model_path] = model
    return model

def predict_yolo(image_path, model_path):
    try:
        model = load_model(model_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图片")
        h, w = img.shape[:2]

        results = model(image_path)
        detections = []

        # 【修改】兼容分类模型和检测模型
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            # 分类模型：获取 top-1 类别
            probs = results[0].probs
            top1 = probs.top1
            confidence = probs.top1conf.item()
            class_id = top1
            # 构造一个检测（使用整图作为 bbox 占位）
            detections.append({
                'bbox': [0.0, 0.0, float(w), float(h)],
                'score': confidence,
                'class_id': class_id
            })
        else:
            # 检测模型：保留原有逻辑（获取所有检测框）
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        score = float(box.conf[0])
                        class_id = int(box.cls[0])
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': score,
                            'class_id': class_id
                        })

        # 补全信息
        for det in detections:
            class_id = det['class_id']
            eng_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Unknown'
            det['class_name'] = map_class_name(eng_name)
            det['advice'] = DISEASE_ADVICE.get(eng_name, '暂无建议')

        # 直接返回原图，不绘制任何框
        annotated_base64 = image_to_base64(img)

        return {
            'success': True,
            'type': 'yolo',
            'detections': detections,
            'image_width': w,
            'image_height': h,
            'annotated_image': annotated_base64,
            'model_used': os.path.basename(model_path),
            'message': f'检测到{len(detections)}个目标'
        }
    except Exception as e:
        return {'success': False, 'detections': [], 'annotated_image': None, 'message': str(e)}
def load_cnn_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # 1. 获取模型架构名称（优先级：checkpoint中的model_name > 文件名解析 > 默认）
    if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        # 从文件名猜测（例如 best_resnet18.pt -> resnet18）
        filename = os.path.basename(model_path).lower()
        if 'resnet18' in filename:
            model_name = 'resnet18'
        elif 'resnet50' in filename:
            model_name = 'resnet50'
        elif 'mobilenet' in filename:
            model_name = 'mobilenet_v2'
        elif 'efficientnet' in filename:
            model_name = 'efficientnet_b0'
        else:
            model_name = 'resnet18'  # 默认
        print(f"[WARN] 从文件名推测模型: {model_name}")

    # 2. 获取类别名称
    if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
        classes = checkpoint['class_names']
    else:
        # 尝试读取同目录 classes.txt 或使用默认
        class_file = os.path.join(os.path.dirname(model_path), 'classes.txt')
        if os.path.exists(class_file):
            with open(class_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            classes = CLASS_NAMES
    num_classes = len(classes)

    # 3. 根据 model_name 动态创建模型
    import torchvision.models as models
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 4. 提取 state_dict 并加载
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, classes


def predict_cnn(image_path, model_path):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, classes = load_cnn_model(model_path,device)
        pil_image = Image.open(image_path).convert('RGB')
        w, h = pil_image.size
        img_cv = cv2.imread(image_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        eng_name = classes[predicted.item()]
        detection = {
            'bbox': [0.0, 0.0, float(w), float(h)],
            'score': confidence.item(),
            'class_id': predicted.item(),
            'class_name': map_class_name(eng_name),
            'advice': DISEASE_ADVICE.get(eng_name, '暂无建议')
        }

        # 直接返回原图
        annotated_base64 = image_to_base64(img_cv)

        return {
            'success': True,
            'type': 'cnn',
            'detections': [detection],
            'image_width': w,
            'image_height': h,
            'annotated_image': annotated_base64,
            'model_used': os.path.basename(model_path),
            'message': '分类完成'
        }
    except Exception as e:
        return {'success': False, 'detections': [], 'annotated_image': None, 'message': str(e)}

def predict_with_image(image_path, model_path, model_type=None):
    if model_type is None:
        model_type = 'cnn' if 'cnn' in model_path.lower() else 'yolo'

    if model_type == 'yolo':
        return predict_yolo(image_path, model_path)
    elif model_type == 'cnn':
        return predict_cnn(image_path, model_path)
    else:
        return {'success': False, 'detections': [], 'annotated_image': None, 'message': '不支持的模型类型'}

def get_class_info():
    chinese_names = [map_class_name(name) for name in CLASS_NAMES]
    return {'names': chinese_names, 'colors': COLORS, 'advice': DISEASE_ADVICE}