import os
import cv2
import numpy as np
import shutil

# ================= 配置区域 =================
# 1. 图片文件夹路径 (对应 labels 文件夹的同级 images 文件夹)
# 假设结构是: datasets2/train/images 和 datasets2/train/labels
base_dataset_path = r"D:\rice-disease-det\pp\datasets\train"
images_dir = os.path.join(base_dataset_path, "images")
labels_dir = os.path.join(base_dataset_path, "labels")

# 2. 输出文件夹
output_dir = r"D:\rice-disease-det\visualized_labels"

# 3. 类别名称列表 (根据你的 data.yaml 中的 names 修改)
# 索引 0 对应 class 0, 索引 1 对应 class 1...
# 如果你的 data.yaml 里是: names: {0: 'blast', 1: 'brown_spot', 2: 'hispa', ...}
# 请在这里按顺序填写，数量要和你的最大 class id + 1 一致
class_names = [
    "Class_0", "Class_1", "Disease_Type_2", "Class_3",
    "Class_4", "Class_5", "Class_6", "Class_7"
]


# ===========================================

def visualize_dataset():
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 获取所有标签文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    if not label_files:
        print(f"❌ 在 {labels_dir} 中未找到 .txt 标签文件！请检查路径。")
        return

    print(f"🚀 开始处理 {len(label_files)} 张图片...")

    count = 0
    for label_file in label_files:
        # 构造对应的图片路径 (支持 jpg, png, jpeg 等)
        img_name_base = os.path.splitext(label_file)[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = os.path.join(images_dir, img_name_base + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if not img_path:
            print(f"⚠️ 跳过：找不到对应的图片文件 {label_file}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳过：无法读取图片 {img_path}")
            continue

        h, w, _ = img.shape

        # 读取标签
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()

        has_valid_box = False

        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3:
                continue

            cls_id = int(parts[0])
            points_flat = parts[1:]

            # 确保点数是偶数 (x, y 成对)
            if len(points_flat) % 2 != 0:
                continue

            coords = np.array(points_flat).reshape(-1, 2)

            # --- 1. 绘制原始多边形 (绿色) ---
            # 将归一化坐标转换为像素坐标
            poly_pixels = coords.copy()
            poly_pixels[:, 0] *= w
            poly_pixels[:, 1] *= h
            poly_pixels = poly_pixels.astype(np.int32)

            # 画多边形轮廓
            cv2.polylines(img, [poly_pixels], isClosed=True, color=(0, 255, 0), thickness=2)
            # 填充半透明多边形 (可选，为了看清内部)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_pixels], color=(0, 255, 0))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # --- 2. 计算并绘制检测框 (红色) ---
            x_min = np.min(coords[:, 0]) * w
            y_min = np.min(coords[:, 1]) * h
            x_max = np.max(coords[:, 0]) * w
            y_max = np.max(coords[:, 1]) * h

            box_width = x_max - x_min
            box_height = y_max - y_min

            # 只有当框有一定大小时才绘制
            if box_width > 1 and box_height > 1:
                has_valid_box = True
                pt1 = (int(x_min), int(y_min))
                pt2 = (int(x_max), int(y_max))

                # 画红色矩形框
                cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=2)

                # 准备标签文字
                label_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                text = f"{label_name}"

                # 计算文字背景大小
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # 画文字背景
                cv2.rectangle(img, (pt1[0], pt1[1] - text_h - baseline - 5),
                              (pt1[0] + text_w, pt1[1]), color=(0, 0, 255), thickness=-1)

                # 画文字
                cv2.putText(img, text, (pt1[0], pt1[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if has_valid_box or len(lines) > 0:
            save_path = os.path.join(output_dir, label_file.replace('.txt', '.jpg'))
            cv2.imwrite(save_path, img)
            count += 1

    print(f"✅ 完成！已生成 {count} 张可视化图片。")
    print(f"📂 查看位置：{output_dir}")
    print("💡 说明：绿色填充区域是原始标注形状，红色框是转换后的检测框。")


if __name__ == '__main__':
    visualize_dataset()