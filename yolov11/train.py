from ultralytics import YOLO  as YOLO
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r'E:\yolov11\ultralytics\cfg\models\addv11\yolov11n-mey.yaml'
#数据集配置文件

data_yaml_path = r'E:\yolov11\data.yaml'
if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=100,
                          batch=8,
                          workers=0,
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/V11train',
                          name='exp',
                          )
