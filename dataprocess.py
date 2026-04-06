import os
import numpy as np

# ================= 配置区域 =================
# 请修改为你的实际标签路径 (建议先用短路径，如 D:\rice_data\datasets2\train\labels)
train_label_dir = r"/datasets2\train\labels"
# 如果有验证集，也请处理
val_label_dir = r"/datasets2\val\labels"
# 如果有测试集，也请处理
test_label_dir = r"/datasets2\test\labels"


dirs_to_process = [train_label_dir, val_label_dir, test_label_dir]


# ===========================================

def clean_segmentation_labels(directory):
    if not os.path.exists(directory):
        print(f"⚠️ 目录不存在：{directory}")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"🧹 开始清洗 {directory} 下的 {len(files)} 个文件...")

    total_lines_removed = 0
    files_modified = 0

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                # 包含非数字字符，丢弃该行
                total_lines_removed += 1
                file_changed = True
                continue

            # 检查坐标点数量
            # 分割任务至少需要3个点 (6个坐标值) 才能构成多边形
            # 如果只有4个坐标值 (2个点)，那是线段，无法生成mask，必须丢弃
            if len(coords) < 6:
                print(f"  ⚠️ 文件 {file_name} 第 {i + 1} 行：点数不足 (仅 {len(coords) // 2} 个点)，已丢弃。")
                total_lines_removed += 1
                file_changed = True
                continue

            # 检查是否为偶数个坐标 (x,y 必须成对)
            if len(coords) % 2 != 0:
                print(f"  ⚠️ 文件 {file_name} 第 {i + 1} 行：坐标数量为奇数，数据损坏，已丢弃。")
                total_lines_removed += 1
                file_changed = True
                continue

            # 额外检查：是否有坐标超出 [0, 1] 范围 (YOLO 归一化要求)
            # 偶尔会有标注工具导出错误，导致坐标 > 1 或 < 0，这也会导致 mask 生成失败
            valid_coords = True
            for c in coords:
                if c < -0.1 or c > 1.1:  # 允许一点点浮点误差
                    valid_coords = False
                    break

            if not valid_coords:
                print(f"  ⚠️ 文件 {file_name} 第 {i + 1} 行：坐标超出归一化范围 [0,1]，已丢弃。")
                total_lines_removed += 1
                file_changed = True
                continue

            # 如果通过所有检查，保留该行
            # 重新格式化以确保一致性
            formatted_line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords) + "\n"
            new_lines.append(formatted_line)

        if file_changed:
            files_modified += 1
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

    print("-" * 30)
    print(f"✅ 清洗完成！")
    print(f"🗑️ 共移除无效行：{total_lines_removed}")
    print(f"📝 共修改文件数：{files_modified}")
    if total_lines_removed == 0 and files_modified == 0:
        print("💡 数据看起来很干净，没有发现明显的格式错误。")
    else:
        print("💡 请重新运行训练脚本，错误应该已解决。")


if __name__ == '__main__':
    for d in dirs_to_process:
        clean_segmentation_labels(d)