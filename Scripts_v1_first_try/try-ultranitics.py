from ultralytics import YOLO

model = YOLO('yolov8x')

path = "C:\\Users\\Slash\\Desktop\\Foot\\input_video\\downloaded_video.mp4"
results = model.predict(path, save=True, stream=True)


ball_detected = 0
person_detected = 0

for result in results:
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name == 'sports ball':
            ball_detected += 1
        if cls_name == 'person':
            person_detected += 1
print('start')
print(f"Frames avec joueurs : {person_detected}")
print(f"Frames avec ballon  : {ball_detected}")
print(f"Classes détectées   : {set(model.names[int(b.cls[0])] for r in model.predict(path, save=True, stream=True) for b in r.boxes)}")