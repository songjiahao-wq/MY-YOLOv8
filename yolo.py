from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）
