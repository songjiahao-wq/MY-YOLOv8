# 训练模型
import cv2
import numpy as np
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")

    # 预测模型
    results = model.predict(source="ultralytics/assets/zidane.jpg", classes=[0], conf=0.5, save=True)[0]  # 0是摄像头，详细参数见3.2
    # boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
    resize_img = cv2.imread('ultralytics/assets/zidane.jpg') # 待裁剪照片
    w, h = resize_img.shape[0], resize_img.shape[1]
    # print(boxes)
    boxes = results.boxes  # Boxes object for bbox outputs[1], box[2], box[3]
    print(boxes.xyxyn)
    for box in np.array(boxes.xyxy.cpu()):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        print(x1, y1, x2, y2)
        resized_img = resize_img[y1:y2, x1:x2]
        cv2.imshow('a', resized_img)
        cv2.waitKey(0)


