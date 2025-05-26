import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QCheckBox, QGroupBox,QShortcut)
from PyQt5.QtCore import Qt, QTimer,QPoint, QSettings
from PyQt5.QtGui import QImage, QPixmap,QBrush,QColor,QPolygon, QPen,QCursor,QKeySequence
import cv2
import numpy as np
from v2 import VideoProcessor,Config
import pygame
import time
from PyQt5.QtGui import QPainter

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("入侵检测系统，郭子靖限定版")
        self.setGeometry(100, 100, 1000, 700)
        
        # 初始化摄像头和处理线程
        self.cap = None
        self.processor = None
        self.is_camera_on = False
        
        # 主界面布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        # 左侧控制面板
        self.control_panel = QGroupBox("控制面板")
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        self.control_panel.setFixedWidth(300)
        
        # 右侧视频显示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        
        # 添加组件到主布局
        self.layout.addWidget(self.control_panel)
        self.layout.addWidget(self.video_label)
        
        # 初始化UI组件
        self.init_ui()
        
        # 定时器用于更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # FPS计算相关变量
        self.frame_count = 0
        self.fps = 0
        self.fps_update_time = time.time()

        self.background_image = None
        self.load_background("background2.jpg")  # 默认背景图片路径

        #全屏
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        
        # 初始状态设为全屏
        self.showFullScreen()

        # # 检测区域相关变量
        # self.drawing_roi = False
        # self.current_roi = []
        # self.roi_points = []
        # self.roi_color = QColor(0, 255, 0, 150)  # 半透明绿色
        
        # # 添加ROI控制按钮
        # self.init_roi_ui()
        
        # # 加载保存的ROI设置
        # self.load_roi_settings()
        
    def init_ui(self):
        """初始化UI组件"""
        # 摄像头控制
        self.camera_btn = QPushButton("开启摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        # 检测类别选择
        self.class_group = QGroupBox("检测类别")
        self.class_layout = QVBoxLayout()
        
        # YOLO常见类别(可根据实际需求调整)
        self.class_checks = []
        classes = [
            ("人", 0),
            ("自行车", 1),
            ("汽车", 2),
            ("摩托车", 3),
            ("飞机", 4),
            ("公交车", 5),
            ("火车", 6),
            ("卡车", 7),
            ("船", 8)
        ]
        
        for name, class_id in classes:
            cb = QCheckBox(name)
            cb.setChecked(class_id in Config.ALARM_CLASSES)
            cb.stateChanged.connect(self.update_alarm_classes)
            self.class_checks.append((cb, class_id))
            self.class_layout.addWidget(cb)
        
        self.class_group.setLayout(self.class_layout)
        
        # 报警声音控制
        self.sound_check = QCheckBox("启用报警声音")
        self.sound_check.setChecked(True)
        
        # 状态信息
        self.status_label = QLabel("状态: 摄像头未开启")
        self.fps_label = QLabel("FPS: 0")
        
        # 添加到控制面板
        self.control_layout.addWidget(self.camera_btn)
        self.control_layout.addWidget(self.class_group)
        self.control_layout.addWidget(self.sound_check)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addWidget(self.fps_label)
        self.control_layout.addStretch()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.setCursor(Qt.ArrowCursor)  # 显示光标
            self.status_label.setText("状态: 窗口模式")
        else:
            self.showFullScreen()
            self.setCursor(Qt.BlankCursor)  # 隐藏光标
            self.status_label.setText("状态: 全屏模式 (按ESC退出)")
            
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Escape:
            # ESC键退出全屏
            if self.isFullScreen():
                self.showNormal()
            else:
                event.ignore()  # 其他情况忽略ESC键
        else:
            super().keyPressEvent(event)
            
    def resizeEvent(self, event):
        """优化全屏下的布局"""
        # 控制面板宽度设为窗口宽度的25%
        panel_width = int(self.width() * 0.25)
        self.control_panel.setFixedWidth(max(300, panel_width))
        
        # 调整字体大小
        font = self.font()
        font.setPointSize(int(self.height() / 50))
        self.setFont(font)
        
        # 更新视频显示
        if self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        
        super().resizeEvent(event)
    

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.is_camera_on:
            self.stop_camera()
            self.camera_btn.setText("开启摄像头")
            self.status_label.setText("状态: 摄像头已关闭")
        else:
            self.start_camera()
            self.camera_btn.setText("关闭摄像头")
            self.status_label.setText("状态: 摄像头运行中")
            
    def start_camera(self):
        """启动摄像头和处理线程"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("状态: 无法打开摄像头")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        # 初始化处理线程
        self.processor = VideoProcessor()
        self.processor.alarm_classes = self.get_selected_classes()
        self.processor.start()
        
        self.is_camera_on = True
        self.timer.start(30)  # 约33FPS
        
    def stop_camera(self):
        """停止摄像头和处理线程"""
        self.timer.stop()
        if self.processor:
            self.processor.stop()
            self.processor = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.is_camera_on = False
        self.video_label.clear()
        
    def update_frame(self):
        """更新视频帧和FPS"""
        if not self.is_camera_on:
            return
            
        start_time = time.time()  # 记录处理开始时间
        
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("状态: 无法读取摄像头帧")
            return
            
        # 将帧送入处理队列
        self.processor.put_frame(frame)
        
        # 获取处理结果
        result = self.processor.get_result()
        if result is not None:
            annotated_frame, alarms = result
            
            # 转换为Qt图像格式
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(), 
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))
            
            # 更新状态
            if alarms:
                self.status_label.setText("状态: 检测到入侵!")
            else:
                self.status_label.setText("状态: 摄像头运行中")
        
        # FPS计算
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.fps_update_time
        
        # 每0.5秒更新一次FPS显示
        if time_diff >= 0.5:
            self.fps = self.frame_count / time_diff
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.fps_update_time = current_time
            
        # 处理耗时计算
        process_time = (time.time() - start_time) * 1000  # 毫秒
        delay = max(1, int(1000 / Config.TARGET_FPS - process_time))
        
        # 确保不会过度延迟
        delay = min(delay, 100)  # 最大延迟100ms
                
    def get_selected_classes(self):
        """获取用户选择的检测类别"""
        selected = []
        for cb, class_id in self.class_checks:
            if cb.isChecked():
                selected.append(class_id)
        return selected
        
    def update_alarm_classes(self):
        """更新报警类别设置"""
        if self.processor:
            self.processor.alarm_classes = self.get_selected_classes()
            
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.stop_camera()
        pygame.mixer.quit()
        event.accept()
    def load_background(self, image_path):
        """后端方法：加载背景图片"""
        try:
            # 使用QPixmap加载背景图片
            self.background_image = QPixmap(image_path)
            if self.background_image.isNull():
                print(f"无法加载背景图片: {image_path}")
                self.background_image = None
        except Exception as e:
            print(f"加载背景图片出错: {e}")
            self.background_image = None
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 绘制背景
        if self.background_image:
            scaled_bg = self.background_image.scaled(
                self.size(), 
                Qt.IgnoreAspectRatio, 
                Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_bg)
        
        # 添加半透明遮罩使内容更清晰
        painter.setBrush(QBrush(QColor(255, 255, 255, 100)))  # 半透明黑色
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        super().paintEvent(event)
        
    def set_background(self, image_path):
        """后端方法：设置新背景（用户界面无此功能）"""
        self.load_background(image_path)
        self.update()  # 触发重绘
    
    # def init_roi_ui(self):
    #         """初始化ROI控制UI """
    #         roi_group = QGroupBox("检测区域设置")
    #         roi_layout = QVBoxLayout()
        
    #         # ROI控制按钮
    #         self.btn_draw_roi = QPushButton("绘制检测区域")
    #         self.btn_draw_roi.clicked.connect(self.start_drawing_roi)
            
    #         self.btn_clear_roi = QPushButton("清除区域")
    #         self.btn_clear_roi.clicked.connect(self.clear_roi)
            
    #         self.btn_save_roi = QPushButton("保存区域")
    #         self.btn_save_roi.clicked.connect(self.save_roi_settings)
            
    #         roi_layout.addWidget(self.btn_draw_roi)
    #         roi_layout.addWidget(self.btn_clear_roi)
    #         roi_layout.addWidget(self.btn_save_roi)
    #         roi_group.setLayout(roi_layout)
            
    #         # 添加到控制面板
    #         self.control_layout.insertWidget(2, roi_group)  # 放在摄像头按钮下方
    
    # def start_drawing_roi(self):
    #     """开始绘制检测区域"""
    #     self.drawing_roi = True
    #     self.current_roi = []
    #     self.status_label.setText("状态: 正在绘制检测区域(点击完成绘制)")
    #     self.btn_draw_roi.setEnabled(False)
    
    # def clear_roi(self):
    #     """清除检测区域"""
    #     self.roi_points = []
    #     self.update()
    #     self.update_config_mask_points()
    
    # def save_roi_settings(self):
    #     """保存ROI设置"""
    #     if len(self.roi_points) < 3:
    #         self.status_label.setText("状态: 需要至少3个点来定义区域")
    #         return
            
    #     settings = QSettings("YourCompany", "DetectionApp_ROI")
    #     points = []
    #     for point in self.roi_points:
    #         # 保存为相对坐标(0-1)
    #         rel_x = point.x() / self.video_label.width()
    #         rel_y = point.y() / self.video_label.height()
    #         points.append((rel_x, rel_y))
        
    #     settings.setValue("roi_points", points)
    #     self.status_label.setText("状态: 检测区域已保存")
        
    #     # 更新配置
    #     self.update_config_mask_points()
    
    # def load_roi_settings(self):
    #     """加载保存的ROI设置"""
    #     settings = QSettings("YourCompany", "DetectionApp_ROI")
    #     points = settings.value("roi_points", [])
        
    #     if points:
    #         self.roi_points = []
    #         for rel_x, rel_y in points:
    #             x = int(float(rel_x) * self.video_label.width())
    #             y = int(float(rel_y) * self.video_label.height())
    #             self.roi_points.append(QPoint(x, y))
            
    #         self.update_config_mask_points()
    
    # def update_config_mask_points(self):
    #     """更新Config中的MASK_POINTS"""
    #     if not self.roi_points:
    #         return
            
    #     # 转换为相对坐标
    #     Config.MASK_POINTS = []
    #     for point in self.roi_points:
    #         rel_x = point.x() / self.video_label.width()
    #         rel_y = point.y() / self.video_label.height()
    #         Config.MASK_POINTS.append((rel_x, rel_y))
        
    #     # 通知处理器更新
    #     if self.processor:
    #         self.processor.update_mask = True
    
    # def mousePressEvent(self, event):
    #     """鼠标点击事件处理"""
    #     if self.drawing_roi and event.button() == Qt.LeftButton:
    #         # 将点击位置转换为视频标签坐标
    #         pos = self.video_label.mapFromParent(event.pos())
    #         if self.video_label.rect().contains(pos):
    #             self.current_roi.append(pos)
    #             self.update()
        
    #     super().mousePressEvent(event)
    
    # def mouseDoubleClickEvent(self, event):
    #     """鼠标双击完成绘制"""
    #     if self.drawing_roi and event.button() == Qt.LeftButton:
    #         if len(self.current_roi) >= 3:
    #             self.roi_points = self.current_roi.copy()
    #             self.drawing_roi = False
    #             self.btn_draw_roi.setEnabled(True)
    #             self.status_label.setText("状态: 检测区域已设置")
    #             self.update_config_mask_points()
    #         else:
    #             self.status_label.setText("状态: 需要至少3个点来定义区域")
        
    #     super().mouseDoubleClickEvent(event)
    
    # def paintEvent(self, event):
    #     """绘制事件"""
    #     super().paintEvent(event)  # 先绘制背景
        
    #     painter = QPainter(self)
    #     pen = QPen(self.roi_color, 2, Qt.SolidLine)
    #     painter.setPen(pen)
    #     painter.setBrush(QBrush(self.roi_color, Qt.Dense4Pattern))
        
    #     # 绘制当前ROI
    #     if self.drawing_roi and self.current_roi:
    #         poly = QPolygon(self.current_roi)
    #         painter.drawPolygon(poly)
            
    #         # 绘制连接线
    #         if len(self.current_roi) > 1:
    #             painter.drawPolyline(poly)
                
    #         # 绘制点到鼠标位置的预览线
    #         mouse_pos = self.video_label.mapFromParent(self.mapFromGlobal(QCursor.pos()))
    #         if self.video_label.rect().contains(mouse_pos):
    #             painter.drawLine(self.current_roi[-1], mouse_pos)
        
    #     # 绘制已保存的ROI
    #     if self.roi_points:
    #         poly = QPolygon(self.roi_points)
    #         painter.drawPolygon(poly)
if __name__ == '__main__':
    # 初始化PyQt应用
    app = QApplication(sys.argv)
    
    # 检查PyGame音频初始化
    pygame.mixer.init()
    
    # 创建并显示主窗口
    window = DetectionApp()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())