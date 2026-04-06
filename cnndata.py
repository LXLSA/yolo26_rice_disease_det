import os
import shutil
from pathlib import Path

# ==================== 配置参数 ====================
YOLO_ROOT = "./datasets2"           # YOLO 数据集根目录，包含 train, val, test 子目录，每个子目录下有 images/ 和 labels/
OUTPUT_ROOT = "./cls_dataset"       # 转换后的分类数据集输出目录
CLASSES_TXT = "./datasets2/classes.txt"  # 类别名称文件（每行一个名称，与 class_id 对应），如果不存在则用 class_id 作为文件夹名
USE_SYMLINK = True                  # 是否使用符号链接（不复制图像，节省空间）
MULTI_CLASS_STRATEGY = "majority"   # 多类别图像处理策略："majority" 取众数，"ignore" 跳过，None 则报错
# =================================================

def read_labels(label_path):
    """读取 YOLO 标注文件，返回类别ID列表"""
    class_ids = []
    if not os.path.exists(label_path):
        return class_ids
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_ids.append(int(parts[0]))
    return class_ids

def get_class_name(class_id, class_names=None):
    """根据 class_id 获取文件夹名"""
    if class_names and class_id < len(class_names):
        return class_names[class_id]
    else:
        return f"class_{class_id}"

def main():
    # 读取类别名称（如果存在）
    class_names = None
    if os.path.exists(CLASSES_TXT):
        with open(CLASSES_TXT, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        print(f"读取到 {len(class_names)} 个类别: {class_names}")
    else:
        print("未找到 classes.txt，将使用 class_ID 作为文件夹名")

    # 处理每个 split
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(YOLO_ROOT, split, 'images')
        labels_dir = os.path.join(YOLO_ROOT, split, 'labels')
        if not os.path.isdir(images_dir):
            print(f"跳过 {split}：{images_dir} 不存在")
            continue

        # 获取所有图片文件（支持常见扩展名）
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"处理 {split} 集，共 {len(img_files)} 张图片")

        for img_file in img_files:
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            # 读取该图像的类别ID列表
            class_ids = read_labels(label_path)
            if not class_ids:
                print(f"警告：{img_file} 无标注，跳过")
                continue

            # 确定图像类别
            unique_ids = set(class_ids)
            if len(unique_ids) == 1:
                target_class = list(unique_ids)[0]
            else:
                # 多类别图像处理
                if MULTI_CLASS_STRATEGY == "majority":
                    # 取出现次数最多的类别
                    target_class = max(set(class_ids), key=class_ids.count)
                    print(f"多类别图像 {img_file}，选择众数类别 {target_class}")
                elif MULTI_CLASS_STRATEGY == "ignore":
                    print(f"多类别图像 {img_file}，跳过")
                    continue
                else:
                    raise ValueError(f"图像 {img_file} 包含多个类别，请设置 MULTI_CLASS_STRATEGY")

            # 构建目标路径
            class_dir = os.path.join(OUTPUT_ROOT, split, get_class_name(target_class, class_names))
            os.makedirs(class_dir, exist_ok=True)
            dst_path = os.path.join(class_dir, img_file)

            # 创建链接或复制
            if USE_SYMLINK:
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(img_path), dst_path)
            else:
                if not os.path.exists(dst_path):
                    shutil.copy2(img_path, dst_path)

        print(f"{split} 集转换完成")

    # 保存类别名称映射（可选）
    if class_names:
        with open(os.path.join(OUTPUT_ROOT, 'classes.txt'), 'w') as f:
            for name in class_names:
                f.write(name + '\n')
    print("转换完成！")

if __name__ == "__main__":
    main()