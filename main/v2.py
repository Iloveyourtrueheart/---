import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
import supervision as sv
import os
from typing import List, Optional, Tuple
import threading
import queue
from collections import deque
from detect_v1 import model_init,annotator,mask_img,predicter,select_alarm_classes

class Config:
    ALARM_CLASSES = [0]  # 默认报警类别(0=person)
    MODEL_PATH = os.path.join('save', 'yolov8n.pt')
    SOUND_FILE = 'alarm.wav'
    CAMERA_WIDTH = 680
    CAMERA_HEIGHT = 480
    COOL_TIME = 5  # 警报冷却时间(秒)
    MAX_QUEUE_SIZE = 3  # 图像队列最大长度
    TARGET_FPS = 30  # 目标帧率
    
    # 监测区域坐标
    MASK_POINTS = [
        (0.1/10, 0.1/10),  # 左上
        (5.5/10, 0.1/10),  # 右上
        (5.5/10, 9.9/10),  # 右下
        (0.1/10, 9.9/10)   # 左下
    ]

class VideoProcessor:
    def __init__(self):
        self.model = model_init(Config.MODEL_PATH)
        self.running = False
        self.frame_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.alarm_status = False
        self.last_alert_time = 0
        self.alarm_classes = Config.ALARM_CLASSES
        
    def start(self):
        """启动处理线程"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """停止处理线程"""
        self.running = False
        self.processing_thread.join()
        
    def put_frame(self, frame: np.ndarray):
        """将帧放入处理队列(非阻塞)"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # 丢弃旧帧，保持最新帧
            
    def get_result(self) -> Optional[Tuple[np.ndarray, List[int]]]:
        """从结果队列获取处理结果(非阻塞)"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
            
    def _process_frames(self):
        """处理线程主函数"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # 处理图像
                masked_img = mask_img(frame.copy(), Config.MASK_POINTS)
                results, processed_img = predicter(self.model, frame, masked_img)
                annotated_frame, alarms = annotator(results, processed_img, self.alarm_classes)
                
                # 检查警报
                current_time = time.time()
                if alarms and (current_time - self.last_alert_time) >= Config.COOL_TIME:
                    self.last_alert_time = current_time
                    self.alarm_status = True
                    trigger(Config.SOUND_FILE)
                
                # 将结果放入队列
                try:
                    self.result_queue.put_nowait((annotated_frame, alarms))
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理帧时出错: {e}")

def trigger(sound_file: str):
    """非阻塞播放警报声音"""
    def play_sound():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
        except Exception as e:
            print(f"播放声音失败: {e}")
    
    audio_thread = threading.Thread(target=play_sound)
    audio_thread.daemon = True
    audio_thread.start()

# 以下是原有的辅助函数(保持不变)
# def model_init(path: str) -> YOLO: ...
# def annotator(results, frame: np.ndarray, alarm_classes: List[int]) -> tuple: ...
# def mask_img(img: np.ndarray, mask_points: List[tuple]) -> np.ndarray: ...
# def predicter(model: YOLO, img: np.ndarray, masked_img: np.ndarray) -> tuple: ...
# def select_alarm_classes() -> List[int]: ...

def main():
    # 初始化
    alarm_classes = select_alarm_classes()
    processor = VideoProcessor()
    processor.alarm_classes = alarm_classes
    processor.start()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    
    # 初始化pygame音频
    pygame.mixer.init()
    
    # FPS计算
    frame_times = deque(maxlen=30)
    last_time = time.time()
    
    try:
        while True:
            # 读取帧
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头帧")
                break
            
            # 将帧送入处理队列
            processor.put_frame(frame)
            
            # 获取处理结果
            result = processor.get_result()
            if result is not None:
                annotated_frame, alarms = result
                
                # 显示结果
                cv2.imshow("入侵检测系统", annotated_frame)
                
                # 显示警报状态
                if alarms:
                    cv2.putText(annotated_frame, "ALARM!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 计算并显示FPS
            current_time = time.time()
            frame_times.append(current_time - last_time)
            last_time = current_time
            if len(frame_times) > 10:
                fps = len(frame_times) / sum(frame_times)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 控制显示速率
            delay = max(1, int(1000 / Config.TARGET_FPS - (time.time() - current_time) * 1000))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
                
    finally:
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == '__main__':
    main()