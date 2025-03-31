from ultralytics import YOLO
# 加载训练好的模型
model = YOLO(r'E:\yolo\ultralytics-main\yolov11\runs\V11train\exp30\weights\best.pt')  # 加载自定义训练模型路经
# 导出模型
model.export(format='onnx')