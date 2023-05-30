from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/models/config/yolov8s-Backbone-ATT.yaml")  # 从头开始构建新模型
    #model.predict('ultralytics/assets', save=True, imgsz=320, conf=0.5)
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）
    #yolo detect train resume model=last.pt
