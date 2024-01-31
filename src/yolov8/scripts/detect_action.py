#!/usr/bin/env python3
import sys
import os
# yoloPath = '/home/nvidia/xjb/algorithm_ws/src/yolov8/'
yoloPath = os.path.dirname(os.path.dirname(__file__))
# print(yoloPath)
sys.path.append(yoloPath) 
from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from yolov8.msg import yoloAction
from yolov8.msg import ActionBox
# import numpy as np
import threading
from action_scripts.detection_keypoint import DetectKeypoint
from action_scripts.classification_keypoint import KeypointClassification

detection_keypoint_model = DetectKeypoint()
classification_keypoint_model = KeypointClassification(yoloPath + '/weights/action/nei_pose_classification.pt')
img = None

def callback(imgmsg):
    global img
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    delay = rospy.Time.now().to_sec() - imgmsg.header.stamp.to_sec()
    print("传图延时=", delay)
    print("---------------------------\n")


def detect_action(yolo_action_result_pub):
    global img
    global detection_keypoint_model
    global classification_keypoint_model
    print('enter detect')
    while True:
        if img is None:
            continue
        results = detection_keypoint_model(img)
        if len(results.keypoints.data.tolist()[0])>10:
            results_keypoint = detection_keypoint_model.get_xy_keypoint(results)
            input_classification = results_keypoint[10:]
            results_classification = classification_keypoint_model(input_classification)

            x_min, y_min, x_max, y_max = results.boxes.xyxy[0].cpu().numpy()
            size = img.shape
            h = size[0] #高度
            w = size[1] #宽度
            x_min, x_max = float(x_min/w), float(x_max/w)
            y_min, y_max = float(y_min/h), float(y_max/h)
            
            # results = model.predict(source=img, half=True, show=False)
            yolo_action_result = yoloAction()
            yolo_action_result.header.stamp = rospy.Time.now()
            yolo_action_Box = ActionBox()
            yolo_action_Box.x1 = x_min
            yolo_action_Box.y1 = y_min
            yolo_action_Box.x2 = x_max
            yolo_action_Box.y2 = y_max
            yolo_action_Box.cls = results_classification.upper()
            yolo_action_result.boxes.append(yolo_action_Box)

            yolo_action_result_pub.publish(yolo_action_result)
            img = None

if __name__ == "__main__":
    rospy.init_node('yolov8_action')
    sub = rospy.Subscriber("/xavier_camera_image", Image, callback)
    yolo_action_result_pub = rospy.Publisher("yolo_action_result", yoloAction, queue_size=1)

    # 创建子线程执行detect函数
    t = threading.Thread(target=detect_action, args=(yolo_action_result_pub,), daemon=True)
    t.start()

    rospy.spin()