#!/usr/bin/python

# 2.12 Lab 4 object detection: a node for detecting objects
# Peter Yu Oct 2016

import rospy
import tf
import numpy as np
import threading
import serial
import pdb
import traceback
import sys
import cv2

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import math
import matplotlib.pyplot as plt

rospy.init_node('object_detection', anonymous=True)
vis_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
cv_bridge = CvBridge()

msg = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None) 
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]

fx = msg.K[0]
fy = msg.K[4]
cx = msg.K[2]
cy = msg.K[5]

def main():
    
    useHSV = False  
    useDepth = False
    if not useHSV:
        # init a cv window
        cv2.namedWindow("image")
        # set callback func for cv window
        cv2.setMouseCallback("image", cvWindowMouseCallBackFunc)
        
        # subscribe to image
        rospy.Subscriber('/camera/rgb/image_color', Image, rosImageVizCallback)
    else:
        if not useDepth:
            # use HSV color conversion to detect the object
            rospy.Subscriber('/camera/rgb/image_color', Image, rosHSVImageVizCallBack)
        else:
            # use both RGB and Depth image
            image_sub = message_filters.Subscriber("camera/rgb/image_color", Image)
            depth_sub = message_filters.Subscriber("camera/depth_registered/image", Image)

            ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.5)
            ts.registerCallback(rosRGBDCallBack)
            
    rospy.spin()
    
    
def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    ## 
    ## xn = ??
    ## yn = ??
    ## xc = ??
    ## yc = ??
    return (xc,yc,zc)
    
    
def showPyramid(xp, yp, zc, w, h):
    X1 = getXYZ(xp-w/2, yp-h/2, zc, fx, fy, cx, cy)
    X2 = getXYZ(xp-w/2, yp+h/2, zc, fx, fy, cx, cy)
    X3 = getXYZ(xp+w/2, yp+h/2, zc, fx, fy, cx, cy)
    X4 = getXYZ(xp+w/2, yp-h/2, zc, fx, fy, cx, cy)
    vis_pub.publish(createConeMarker(1, [X1,X2,X3,X4], [1,0,0,1], frame_id = '/camera'))

def depthImageCallBack(msg):
    # 1. convert ROS image to opencv format
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "32FC1")
    except CvBridgeError as e:
        print(e)
    
def cvWindowMouseCallBackFunc(event, xp, yp, flags, param):
    print 'In cvWindowMouseCallBackFunc: (xp, yp)=', xp, yp 
    zc = 2.0
    # visualize the pyramid
    showPyramid(xp, yp, zc, 10, 10)

def createConeMarker(marker_id, points, rgba, frame_id = '/camera'):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = marker.TRIANGLE_LIST
    marker.scale = Vector3(1,1,1)
    marker.id = marker_id
    
    n = len(points)
    
    if rgba is not None:
        marker.color = ColorRGBA(*rgba)
        
    
    o = Point(0,0,0)
    for i in xrange(n):
        p = Point(*points[i])
        marker.points.append(p)
        p = Point(*points[(i+1)%4])
        marker.points.append(p)
        marker.points.append(o)
        
    marker.pose = poselist2pose([0,0,0,0,0,0,1])
    return marker

def poselist2pose(poselist):
    return Pose(Point(*poselist[0:3]), Quaternion(*poselist[3:7]))

def rosImageVizCallback(msg):
    # 1. convert ROS image to opencv format
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    # 2. visualize it in a cv window
    cv2.imshow("image", cv_image)
    cv2.waitKey(3)

    
def rosHSVImageVizCallBack(msg):
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV (Change the thresholds here)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    mask_eroded         = cv2.erode(mask, None,iterations = 3)
    mask_eroded_dilated = cv2.dilate(mask_eroded, None,iterations = 10)
    
    print 'hsv', hsv[240][320]   # this printout is useful to find out the threshold

    # Find the blobs in the mask and get their (xp,yp)
    contours,hierarchy = cv2.findContours(mask_eroded_dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        xp,yp,w,h = cv2.boundingRect(cnt)
        zc = 2 # guess
        centerx,centery = xp+w/2, yp+h/2
        cv2.rectangle(cv_image,(xp,yp),(xp+w,yp+h),[0,255,255],2)
        
        showPyramid(centerx, centery, zc, w, h)
    

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(cv_image,cv_image, mask= mask_eroded_dilated)
    cv2.line(cv_image, (320, 235), (320, 245), (255,0,0))
    cv2.line(cv_image, (325, 240), (315, 240), (255,0,0))
    cv2.imshow('image',cv_image)
    cv2.imshow('mask',mask_eroded_dilated)
    cv2.imshow('res',res)
    cv2.waitKey(3)

def rosRGBDCallBack(rgb_data, depth_data):
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        cv_depthimage = cv_bridge.imgmsg_to_cv2(depth_data, "32FC1")
        cv_depthimage2 = np.array(cv_depthimage, dtype=np.float32)
    except CvBridgeError as e:
        print(e)
        
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV (Change the thresholds here as in Task 2)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    mask_eroded         = cv2.erode(mask, None,iterations = 3)
    mask_eroded_dilated = cv2.dilate(mask_eroded, None,iterations = 10)
    
    contours,hierarchy = cv2.findContours(mask_eroded_dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        xp,yp,w,h = cv2.boundingRect(cnt)
        
        # get zc from depth image
        if not math.isnan(cv_depthimage2[int(yp)][int(xp)] ) and cv_depthimage2[int(yp)][int(xp)] > 0.1:
            zc = cv_depthimage2[int(yp)][int(xp)]
            print zc
        else:
            continue
            zc = 1
            
        print cv_depthimage2[int(yp)][int(xp)]
        centerx,centery = xp+w/2, yp+h/2
        cv2.rectangle(cv_image,(xp,yp),(xp+w,yp+h),[0,255,255],2)
        
        showPyramid(centerx, centery, zc, w, h)
            
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(cv_image,cv_image, mask= mask_eroded_dilated)
    cv2.line(cv_image, (320, 235), (320, 245), (255,0,0))
    cv2.line(cv_image, (325, 240), (315, 240), (255,0,0))
    cv2.imshow('image',cv_image)
    cv2.imshow('mask',mask_eroded_dilated)
    cv2.imshow('res',res)
    
    cv2.waitKey(3)
    
if __name__=='__main__':
    main()
    
