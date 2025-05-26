import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
import supervision as sv
import os
from typing import List

# 配置参数
class Config:
    # 默认报警类别（对应YOLO类别ID），0=person, 1=bicycle, 2=car等
    ALARM_CLASSES = [0]  
    MODEL_PATH = os.path.join('save', 'yolov8n.pt')
    SOUND_FILE = 'You.mp3'
    CAMERA_WIDTH = 680
    CAMERA_HEIGHT = 480
    COOL_TIME = 5  # 警报冷却时间(秒)
    
    # 监测区域坐标(相对于图像大小的比例)
    MASK_POINTS = [
        (0.1/10, 0.1/10),  # 左上
        (5.5/10, 0.1/10),  # 右上
        (5.5/10, 9.9/10),  # 右下
        (0.1/10, 9.9/10)   # 左下
    ]

def model_init(model_path: str) -> YOLO:
    """初始化YOLO模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def annotator(results, frame: np.ndarray, alarm_classes: List[int]) -> tuple:
    """
    标注检测结果
    :param results: YOLO检测结果
    :param frame: 原始帧
    :param alarm_classes: 需要报警的类别列表
    :return: 标注后的帧, 检测到的报警类别列表
    """
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.default(), 
        thickness=3, 
        text_thickness=3,
        text_scale=1
    )
    
    alarm_detected = []
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id in alarm_classes:
            alarm_detected.append(class_id)
    
    detections = sv.Detections.from_ultralytics(results[0])
    frame = box_annotator.annotate(scene=frame, detections=detections)
    return frame, alarm_detected

def trigger(sound_file: str) -> None:
    """播放警报声音"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        time.sleep(2)
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # 释放资源
    except Exception as e:
        print(f"播放声音失败: {e}")

def mask_img(img: np.ndarray, mask_points: List[tuple]) -> np.ndarray:
    """
    创建监测区域掩码
    :param img: 原始图像
    :param mask_points: 监测区域坐标比例列表[(x1,y1), (x2,y2), ...]
    :return: 掩码后的图像
    """
    mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    
    # 将比例坐标转换为实际像素坐标
    pts = np.array([
        [img.shape[1] * x, img.shape[0] * y] for x, y in mask_points
    ], np.int32).reshape((-1, 1, 2))
    
    mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    masked_img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask) 
    return masked_img

def predicter(model: YOLO, img: np.ndarray, masked_img: np.ndarray) -> tuple:
    """
    进行目标检测
    :param model: YOLO模型
    :param img: 原始图像
    :param masked_img: 掩码后的图像
    :return: 检测结果, 处理后的图像
    """
    results = model(masked_img, stream=False, save=False, imgsz=640)
    
    # 打印检测到的类别
    detected_classes = results[0].boxes.cls.tolist()
    print(f"检测到的类别: {detected_classes}")
    
    # 可视化处理
    sub_img = cv2.subtract(img, masked_img)
    masked_img_detected = results[0].plot()
    result_img = cv2.add(masked_img_detected, sub_img)
    
    return results, result_img

def select_alarm_classes() -> List[int]:
    """让用户选择需要报警的类别"""
    print("请选择需要报警的类别(输入数字，多个用逗号分隔):")
    print("0: person, 1: bicycle, 2: car, 3: motorcycle, ...")
    choices = input("输入选择的类别(例如: 0,1,2): ").strip()
    
    try:
        alarm_classes = [int(c.strip()) for c in choices.split(',') if c.strip()]
        print(f"已选择报警类别: {alarm_classes}")
        return alarm_classes
    except ValueError:
        print("输入无效，将使用默认报警类别(0: person)")
        return Config.ALARM_CLASSES

def main():
    # 让用户选择报警类别
    alarm_classes = select_alarm_classes()
    
    # 初始化模型
    model = model_init(Config.MODEL_PATH)
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    
    # 初始化pygame音频
    pygame.mixer.init()
    
    last_alert_time = 0  # 上次报警时间
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头帧")
                break
            
            # 应用监测区域掩码
            masked_img = mask_img(frame.copy(), Config.MASK_POINTS)
            
            # 进行目标检测
            results, processed_img = predicter(model, frame, masked_img)
            
            # 标注结果
            annotated_frame, alarms = annotator(results, processed_img, alarm_classes)
            
            # 显示结果
            cv2.imshow("入侵检测系统", annotated_frame)
            
            # 检查是否需要触发警报
            if alarms and (time.time() - last_alert_time) >= Config.COOL_TIME:
                trigger(Config.SOUND_FILE)
                last_alert_time = time.time()
                print(f"警报触发! 检测到类别: {alarms}")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == '__main__':
    main()