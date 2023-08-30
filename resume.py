from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    # 断点续练
    # model = YOLO(r"F:\sjh\code\ultralytics\runs\detect\cfg\weights\last.pt")
    # model.train(resume=True)

    # 加载模型
    model = YOLO('runs/detect/yolov8n-SPPCSPC/weights/last.pt')  # 从头开始构建新模型
    # Use the model
    results = model.train(resume=True)  # 训练模型
    # results = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

