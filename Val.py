from ultralytics import YOLO
if __name__ == '__main__':
    # model = YOLO("yolov8s.yaml")  # 从头开始构建新模型
    model = YOLO(r"D:\yanyi\project_process\ultralytics\runs\detect\train2\weights/best.pt")  # 加载预训练模型（推荐用于训练）
    results = model.val(data="NEU.yaml",batch=1)  # 在验证集上评估模型性能