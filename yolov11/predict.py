from ultralytics import YOLO
# model = YOLO(r"yolov11n.pt")
# model = YOLO(r"E:\yolo\ultralytics-main\yolov11\runs\V11train\exp\weights\best.pt")
# results = model.predict(r"E:\yolo\ultralytics-main\yolov11\ultralytics\assets\bus.jpg")
# results[0].show()
# results[0].save("output_image.jpg")

if __name__ == '__main__':
    model = YOLO(r'E:\yolo\ultralytics-main\yolov11\runs\V11train\exp3\weights\best.pt')
    model.predict(source=r'E:\yolo\ultralytics-main\yolov11\datasets\data1\images\val',
                  project='runs/detect',
                  name='exp',
                  save=True,
                )