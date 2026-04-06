from ultralytics import YOLO
import os

# ==================== 配置参数 ====================
# 修改为分类模型权重 (cls = classification)
# 可选：'yolov8n-cls.pt', 'yolo11n-cls.pt', 'yolo26n-cls.pt'
yolo_name = 'yolov8n-cls.pt'

# 数据集配置文件路径
# 注意：分类任务的 data.yaml 格式通常为:
# train: path/to/train
# val: path/to/val
# test: path/to/test (可选)
# names: [class1, class2, ...]
DATA_YAML = r"D:\rice-disease-det\datasets"

# 通用训练参数
IMGSZ = 224  # 分类任务通常使用 224 或 640，224 速度更快
WORKERS = 0  # Windows建议设为0
BATCH = 16

# 是否执行后续步骤
RUN_VAL = True
RUN_TEST = True
EXPORT_ONNX = False
RUN_INFERENCE = True


# =================================================

def main():
    model = None

    # 1. 初始化模型并训练
    if yolo_name == 'yolov8n-cls.pt':
        model = YOLO("yolov8n-cls.pt")
        results = model.train(
            data=DATA_YAML,
            epochs=200,  # 示例用短轮数
            imgsz=IMGSZ,
            workers=WORKERS,
            batch=BATCH,
            project='runs/classify/yolov8n-cls',  # 指定项目目录
            name='yolov8n-cls-exp'  # 指定实验名称
        )
    elif yolo_name == 'yolo11n-cls.pt':
        model = YOLO("yolo11n-cls.pt")
        results = model.train(
            data=DATA_YAML,
            epochs=20,
            imgsz=IMGSZ,
            workers=WORKERS,
            batch=BATCH,
            project='runs/classify/yolo11n-cls',
            name='yolo11n-cls-exp'
        )
    elif yolo_name == 'yolo26n-cls.pt':
        model = YOLO("yolo26n-cls.pt")
        results = model.train(
            data=DATA_YAML,
            project='runs/classify/yolo26n-cls',
            epochs=200,  # 完整训练轮数
            imgsz=IMGSZ,
            seed=42,
            deterministic=True,
            workers=WORKERS,
            batch=BATCH
        )
    else:
        raise ValueError(f"未知的模型名称: {yolo_name}")

    # 确保模型对象可用（如果是 resume 模式，上面可能没赋值，这里补全逻辑）
    if model is None and hasattr(results, 'model'):
        model = results.model

    # 2. 训练完成后，在验证集上评估
    if RUN_VAL and model:
        print("\n===== 开始在验证集上评估 (Classification) =====")
        # 分类任务的 val 返回 top1_acc, top5_acc 等
        val_results = model.val()
        print("验证集指标:", val_results.results_dict)

    # 3. 在测试集上评估
    if RUN_TEST and model:
        print("\n===== 开始在测试集上评估 =====")
        try:
            # 需要 data.yaml 中有 test 字段
            test_results = model.val(split='test')
            print("测试集指标:", test_results.results_dict)
        except Exception as e:
            print(f"测试集评估失败（可能未定义 test 路径）: {e}")

    # # 4. 导出模型为 ONNX
    # if EXPORT_ONNX and model:
    #     print("\n===== 导出 ONNX 模型 =====")
    #     success = model.export(format='onnx', imgsz=IMGSZ)
    #     if success:
    #         print("ONNX 导出成功，保存路径:", success)
    #     else:
    #         print("ONNX 导出失败")

    # 5. 单张图片推理示例
    if RUN_INFERENCE and model:
        print("\n===== 单张图片推理示例 (Classification) =====")
        # 请替换为实际的图片路径
        test_image = r"D:\rice-disease-det\static\images\daowenbing.jpg"

        if os.path.exists(test_image):
            # 分类任务的预测结果
            pred = model(test_image, save=True)

            print("推理完成，结果保存在 runs/classify/predict 目录下")

            # 分类任务的结果解析方式与分割/检测不同
            # pred[0].probs 包含概率分布
            probs = pred[0].probs

            # 获取置信度最高的类别索引和置信度
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            top5_idx = probs.top5  # 前5个类别的索引

            # 获取类别名称 (需要从模型数据中获取，或者手动映射)
            # model.names 是一个字典 {0: 'class_name', ...}
            class_name = model.names[top1_idx]

            print(f"预测结果:")
            print(f"  - 最可能的类别: {class_name}")
            print(f"  - 置信度: {top1_conf:.4f}")
            print(f"  - Top 5 类别索引: {top5_idx}")

            # 如果需要打印所有类别的概率
            # for i, conf in enumerate(probs.data):
            #     print(f"  {model.names[i]}: {conf.item():.4f}")

        else:
            print(f"测试图片 {test_image} 不存在，跳过推理示例")


if __name__ == '__main__':
    # 选项 A: 从头开始训练 (取消注释下面这行)
    main()
    # 选项 B: 断点续训 (Resume) - 针对分类模型
    # 请确保路径指向的是分类任务的 weights (last.pt 或 best.pt)
    checkpoint_path = r"D:\rice-disease-det\models\runs\classify\runs\classify\yolo26n-cls\train3\weights\last.pt"
    if os.path.exists(checkpoint_path):
        print(f"正在加载检查点进行续训: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        # resume=True 会自动读取之前的配置 (epochs, imgsz, data 等) 继续训练
        results = model.train(resume=True)
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
        print("将运行一次完整的训练示例...")
        # 如果文件不存在，运行一次主函数作为演示
        # main()