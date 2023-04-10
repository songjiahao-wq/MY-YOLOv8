from ultralytics import YOLO
if __name__ == '__main__':
    # 断点续练
    # model = YOLO(r"F:\sjh\code\ultralytics\runs\detect\cfg\weights\last.pt")
    # model.train(resume=True)

    # 加载模型
    model = YOLO("yolov8s-Bottleneck-ATT.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）

    # Use the model
    results = model.train(data="VOC_five.yaml", epochs=150, batch=16, workers=8, close_mosaic=0, name='yolov8s-Bottleneck-ATT-exp1')  # 训练模型
    # results = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式