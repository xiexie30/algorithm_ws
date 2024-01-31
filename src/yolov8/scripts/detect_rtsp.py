#!/usr/bin/env python3
import sys
import os
yoloPath = '/home/nvidia/xjb/algorithm_ws/src/yolov8/'
# 获取当前文件所在路径
print(os.path.abspath(__file__))
sys.path.append(yoloPath) 
from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from yolov8.msg import yolo
from yolov8.msg import Box
# import numpy as np
import threading
from action_scripts.detection_keypoint import DetectKeypoint
from action_scripts.classification_keypoint import KeypointClassification

detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('weights/action/nei_pose_3_classification.pt')

# 打开视频文件
video = cv2.VideoCapture(0)

# 检查视频是否成功打开
if not video.isOpened():
    print("无法打开摄像头")
    exit()

# 获取视频帧的宽度和高度
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频对象
# output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

while True:
    # 读取视频帧
    ret, image = video.read()

    # 如果视频帧读取失败，则退出循环
    if not ret:
        break
        
    results = detection_keypoint(image)

    if len(results.keypoints.data.tolist()[0])>10:
        results_keypoint = detection_keypoint.get_xy_keypoint(results)
        input_classification = results_keypoint[10:] 
        results_classification = classification_keypoint(input_classification)

        height, width = image.shape[:2]

        image_draw = results.plot(boxes=False)

        x_min, y_min, x_max, y_max = results.boxes.xyxy[0].cpu().numpy()
        image_draw = cv2.rectangle(
                        image_draw, 
                        (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                        (0,0,255), 2
                    )
        (w, h), _ = cv2.getTextSize(
                results_classification.upper(), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
        image_draw = cv2.rectangle(
                        image_draw, 
                        (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)), 
                        (0,0,255), -1
                    )
        cv2.putText(image_draw,
                    f'{results_classification.upper()}',
                    (int(x_min), int(y_min)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255),
                    thickness=2
                )
        # 在输出视频中写入帧
        # output_video.write(image_draw)

        # 显示结果帧
    else:
        image_draw=image
    cv2.imshow('Frame', image_draw)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video.release()
# output_video.release()
cv2.destroyAllWindows()
