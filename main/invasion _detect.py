import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from PIL import Image
import time
import supervision as sv
alarm_list = [0]#报警的类别
path_weights = 'save\\yolov8n.pt'
def model_init(path_weights):
    model = YOLO(path_weights)
    return model
def annotator(results,fram):
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=3, text_thickness=3,text_scale=1)
    xyxys = []
    confidences = []
    class_ids = []
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id in alarm_list:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy())
    detections = sv.Detections.from_ultralytics(results[0])
    fram = box_annotator.annotate(scene=fram,detections=detections)
    return fram,class_ids

def trigger(sound_file):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    time.sleep(2)
    pygame.mixer.music.stop()
def check_trigger_state(list1,list2):
     return any(item in list2 for item in list1)

# 原始图像和掩码
def mask_img(img):
    hl1 = 0.1 / 10 #监测区域高度距离图片顶部比例
    wl1 = 0.1 / 10 #监测区域高度距离图片左部比例
    hl2 = 0.1 / 10  # 监测区域高度距离图片顶部比例
    wl2 = 5.5 / 10  # 监测区域高度距离图片左部比例
    hl3 = 9.9 / 10  # 监测区域高度距离图片顶部比例
    wl3 = 5.5 / 10  # 监测区域高度距离图片左部比例
    hl4 = 9.9 / 10  # 监测区域高度距离图片顶部比例
    wl4 = 0.1 / 10  # 监测区域高度距离图片左部比例
    mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    pts = np.array([[img.shape[1] * wl1, img.shape[0] * hl1],  # pts1
                    [img.shape[1] * wl2, img.shape[0] * hl2],  # pts2
                    [img.shape[1] * wl3, img.shape[0] * hl3],  # pts3
                    [img.shape[1] * wl4, img.shape[0] * hl4]], np.int32).reshape((-1,1,2))
    mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    masked_img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask) 
    return masked_img   
def predicter(model,img):
    results = model(img, stream=False,save=False,imgsz=640) 
    print(results[0].boxes.cls.tolist())
    sub_img = cv2.subtract(img, masked_img)
    masked_img_detected = results[0].plot()
    result0 = cv2.add(masked_img_detected,sub_img)
    return results
if __name__ == '__main__':
    model = model_init(path_weights)
    #trigger_state = False
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cool_time = 5
    last_alert_time = time.time()
    while True:
        success, frame = cap.read()
        if success:
            masked_img = mask_img(frame)
            results = predicter(model,masked_img)
            fram,alarm = annotator(results,frame)
            
            cv2.imshow("YOLOv8 Inference", fram)
                     
            if len(alarm) > 0:
                current_time = time.time()
                time_since_last_alert = current_time - last_alert_time
                if time_since_last_alert >= cool_time:
                    trigger('You.mp3')
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



