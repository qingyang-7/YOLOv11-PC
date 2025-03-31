from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r"E:\yolo\ultralytics-main\yolov11\runs\V11train\exp\weights\best.pt"
#数据集配置文件
data_yaml_path = r'E:\yolo\ultralytics-main\yolov11\datasets\data.yaml'

if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    model.val(data=data_yaml_path,
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              project='runs/val',
              name='exp',
              # save_json = True
              )