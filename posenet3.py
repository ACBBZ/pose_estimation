#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
import re
import cv2
import numpy
import math

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)


input = jetson.utils.videoSource("./75.jpg", argv=sys.argv)
img = input.Capture()
poses = net.Process(img, overlay=opt.overlay)

img=numpy.array(img)
b,g,r = cv2.split(img) 
img = cv2.merge([r,g,b])
cv2.imshow('Pre',img)

cv2.waitKey(1)

for pose in poses:
    #print(pose)
    print(pose.Keypoints)
    #print('Links', pose.Links)

    left_shoulder_idx = pose.FindKeypoint('left_shoulder')
    right_shoulder_idx = pose.FindKeypoint('right_shoulder')
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    right_wrist_idx = pose.FindKeypoint('right_wrist')
    left_knee_idx = pose.FindKeypoint('left_knee')
    right_knee_idx = pose.FindKeypoint('right_knee')
    left_ankle_idx = pose.FindKeypoint('left_ankle')
    right_ankle_idx = pose.FindKeypoint('right_ankle')
    left_elbow_idx = pose.FindKeypoint('left_elbow')
    right_elbow_idx = pose.FindKeypoint('right_elbow')
    left_hip_idx = pose.FindKeypoint('left_hip')
    right_hip_idx = pose.FindKeypoint('right_hip')

    # if the keypoint index is < 0, it means it wasn't found in the image
    if right_shoulder_idx > 0 and left_shoulder_idx > 0:
        pre_left_shoulder = pose.Keypoints[left_shoulder_idx]
        pre_right_shoulder = pose.Keypoints[right_shoulder_idx]
        pre_dy_shoulder = abs(pre_right_shoulder.y - pre_left_shoulder.y)
        pre_dx_shoulder = abs(pre_right_shoulder.x - pre_left_shoulder.x)
        pre_dz_2_shoulder = pre_dy_shoulder ** 2 + pre_dx_shoulder ** 2
        pre_dz_shoulder = math.sqrt(pre_dz_2_shoulder) 
        pre_angle1_shoulder = math.acos(pre_dx_shoulder / pre_dz_shoulder)
        pre_angle1_shoulder = pre_angle1_shoulder * 180 / math.pi	
        print("pre_action_shoulder",pre_angle1_shoulder)

    if right_elbow_idx > 0 and right_shoulder_idx > 0:  
        pre_right_elbow = pose.Keypoints[right_elbow_idx]
        pre_right_shoulder = pose.Keypoints[right_shoulder_idx]
        pre_dy_right_elbow = abs(pre_right_shoulder.y - pre_right_elbow.y)
        pre_dx_right_elbow = abs(pre_right_shoulder.x - pre_right_elbow.x)
        pre_dz_2_right_elbow = pre_dy_right_elbow ** 2 + pre_dx_right_elbow ** 2
        pre_dz_right_elbow = math.sqrt(pre_dz_2_right_elbow) 
        pre_angle1_right_elbow = math.acos(pre_dx_right_elbow / pre_dz_right_elbow)
        pre_angle1_right_elbow = pre_angle1_right_elbow * 180 / math.pi
        print("pre_action_right_elbow",pre_angle1_right_elbow)

    if left_elbow_idx > 0 and left_shoulder_idx > 0:  
        pre_left_elbow = pose.Keypoints[left_elbow_idx]
        pre_left_shoulder = pose.Keypoints[left_shoulder_idx]
        pre_dy_left_elbow = abs(pre_left_shoulder.y - pre_left_elbow.y)
        pre_dx_left_elbow = abs(pre_left_shoulder.x - pre_left_elbow.x)
        pre_dz_2_left_elbow = pre_dy_left_elbow ** 2 + pre_dx_left_elbow ** 2
        pre_dz_left_elbow = math.sqrt(pre_dz_2_left_elbow) 
        pre_angle1_left_elbow = math.acos(pre_dx_left_elbow / pre_dz_left_elbow)
        pre_angle1_left_elbow = pre_angle1_left_elbow * 180 / math.pi
        print("pre_action_left_elbow",pre_angle1_left_elbow)

    if right_elbow_idx > 0 and right_wrist_idx > 0:   
        pre_right_elbow = pose.Keypoints[right_elbow_idx]
        pre_right_wrist = pose.Keypoints[right_wrist_idx]
        pre_dy_right_wrist = abs(pre_right_elbow.y - pre_right_wrist.y)
        pre_dx_right_wrist = abs(pre_right_elbow.x - pre_right_wrist.x)
        pre_dz_2_right_wrist = pre_dy_right_wrist ** 2 + pre_dx_right_wrist ** 2
        pre_dz_right_wrist = math.sqrt(pre_dz_2_right_wrist) 
        pre_angle1_right_wrist = math.acos(pre_dx_right_wrist / pre_dz_right_wrist)
        pre_angle1_right_wrist = pre_angle1_right_wrist * 180 / math.pi
        print("pre_action_right_wrist",pre_angle1_right_wrist)

    if left_elbow_idx > 0 and left_wrist_idx > 0:  
        pre_left_elbow = pose.Keypoints[left_elbow_idx]
        pre_left_wrist = pose.Keypoints[left_wrist_idx]
        pre_dy_left_wrist = abs(pre_left_elbow.y - pre_left_wrist.y)
        pre_dx_left_wrist = abs(pre_left_elbow.x - pre_left_wrist.x)
        pre_dz_2_left_wrist = pre_dy_left_wrist ** 2 + pre_dx_left_wrist ** 2
        pre_dz_left_wrist = math.sqrt(pre_dz_2_left_wrist) 
        pre_angle1_left_wrist = math.acos(pre_dx_left_wrist / pre_dz_left_wrist)
        pre_angle1_left_wrist = pre_angle1_left_wrist * 180 / math.pi
        print("pre_action_left_wrist",pre_angle1_left_wrist)

    if right_knee_idx > 0 and right_ankle_idx > 0:   
        pre_right_knee = pose.Keypoints[right_knee_idx]
        pre_right_ankle = pose.Keypoints[right_ankle_idx]
        pre_dy_right_leg = abs(pre_right_knee.y - pre_right_ankle.y)
        pre_dx_right_leg = abs(pre_right_knee.x - pre_right_ankle.x)
        pre_dz_2_right_leg = pre_dy_right_leg ** 2 + pre_dx_right_leg ** 2
        pre_dz_right_leg = math.sqrt(pre_dz_2_right_leg) 
        pre_angle1_right_leg = math.acos(pre_dx_right_leg / pre_dz_right_leg)
        pre_angle1_right_leg = pre_angle1_right_leg * 180 / math.pi
        print("pre_action_right_leg",pre_angle1_right_leg)

    if left_knee_idx > 0 and left_ankle_idx > 0:   
        pre_left_knee = pose.Keypoints[left_knee_idx]
        pre_left_ankle = pose.Keypoints[left_ankle_idx]
        pre_dy_left_leg = abs(pre_left_knee.y - pre_left_ankle.y)
        pre_dx_left_leg = abs(pre_left_knee.x - pre_left_ankle.x)
        pre_dz_2_left_leg = pre_dy_left_leg ** 2 + pre_dx_left_leg ** 2
        pre_dz_left_leg = math.sqrt(pre_dz_2_left_leg) 
        pre_angle1_left_leg = math.acos(pre_dx_left_leg / pre_dz_left_leg)
        pre_angle1_left_leg = pre_angle1_left_leg * 180 / math.pi
        print("pre_action_left_leg",pre_angle1_left_leg)

    if right_knee_idx > 0 and right_hip_idx > 0:   
        pre_right_knee = pose.Keypoints[right_knee_idx]
        pre_right_hip = pose.Keypoints[right_hip_idx]
        pre_dy_right_thigh = abs(pre_right_knee.y - pre_right_hip.y)
        pre_dx_right_thigh = abs(pre_right_knee.x - pre_right_hip.x)
        pre_dz_2_right_thigh = pre_dy_right_thigh ** 2 + pre_dx_right_thigh ** 2
        pre_dz_right_thigh = math.sqrt(pre_dz_2_right_thigh) 
        pre_angle1_right_thigh = math.acos(pre_dx_right_thigh / pre_dz_right_thigh)
        pre_angle1_right_thigh = pre_angle1_right_thigh * 180 / math.pi
        print("pre_action_right_thigh",pre_angle1_right_thigh)

    if left_knee_idx > 0 and left_hip_idx > 0:   
        pre_left_knee = pose.Keypoints[left_knee_idx]
        pre_left_hip = pose.Keypoints[left_hip_idx]
        pre_dy_left_thigh = abs(pre_left_knee.y - pre_left_hip.y)
        pre_dx_left_thigh = abs(pre_left_knee.x - pre_left_hip.x)
        pre_dz_2_left_thigh = pre_dy_left_thigh ** 2 + pre_dx_left_thigh ** 2
        pre_dz_left_thigh = math.sqrt(pre_dz_2_left_thigh) 
        pre_angle1_left_thigh = math.acos(pre_dx_left_thigh / pre_dz_left_thigh)
        pre_angle1_left_thigh = pre_angle1_left_thigh * 180 / math.pi
        print("pre_action_left_thigh",pre_angle1_left_thigh)

    left_shoulder = pose.Keypoints[left_shoulder_idx]
    right_shoulder = pose.Keypoints[right_shoulder_idx]
    left_wrist = pose.Keypoints[left_wrist_idx]
    right_wrist = pose.Keypoints[left_wrist_idx]
    left_knee = pose.Keypoints[left_knee_idx]
    right_knee = pose.Keypoints[left_knee_idx]
    left_ankle = pose.Keypoints[left_ankle_idx]
    right_ankle = pose.Keypoints[left_ankle_idx]
    left_elbow = pose.Keypoints[left_elbow_idx]
    right_elbow = pose.Keypoints[left_elbow_idx]
    left_hip = pose.Keypoints[left_hip_idx]
    right_hip = pose.Keypoints[left_hip_idx]

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
#input = jetson.utils.videoSource(opt.input_URI, argv=["--input-flip=rotate-180"])
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)
    img=numpy.array(img)
    # print the pose results
    #print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        #print(pose.Keypoints)
        #print('Links', pose.Links)

        left_shoulder_idx = pose.FindKeypoint('left_shoulder')
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        right_wrist_idx = pose.FindKeypoint('right_wrist')
        left_knee_idx = pose.FindKeypoint('left_knee')
        right_knee_idx = pose.FindKeypoint('right_knee')
        left_ankle_idx = pose.FindKeypoint('left_ankle')
        right_ankle_idx = pose.FindKeypoint('right_ankle')
        left_elbow_idx = pose.FindKeypoint('left_elbow')
        right_elbow_idx = pose.FindKeypoint('right_elbow')
        left_hip_idx = pose.FindKeypoint('left_hip')
        right_hip_idx = pose.FindKeypoint('right_hip')


        if pose.ID == 0:
            box_left = round(pose.left)
            box_top = round(cur_id_top-50)
            box_right = round(cur_id_right)
            box_bottom = round(cur_id_bottom)

            cv2.rectangle(img,(box_left,box_top),(box_right,box_bottom),(0,0,255),2)

        # if the keypoint index is < 0, it means it wasn't found in the image
            if right_shoulder_idx > 0 and left_shoulder_idx > 0:
                left_shoulder = pose.Keypoints[left_shoulder_idx]
                right_shoulder = pose.Keypoints[right_shoulder_idx]
                dy_shoulder = abs(right_shoulder.y - left_shoulder.y)
                dx_shoulder = abs(right_shoulder.x - left_shoulder.x)
                dz_2_shoulder = dy_shoulder ** 2 + dx_shoulder ** 2
                dz_shoulder = math.sqrt(dz_2_shoulder) 
                angle1_shoulder = math.acos(dx_shoulder / dz_shoulder)
                angle1_shoulder = angle1_shoulder * 180 / math.pi

                if abs(angle1_shoulder - pre_angle1_shoulder) <= 10:
                        #a="shoulders level"
                    a=""
                else:
                    a="ID 0 shoulders need to be level"
                #print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
                cv2.putText(img,a,(50,200),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),3,cv2.LINE_AA)
    


    b,g,r = cv2.split(img) 
    img = cv2.merge([r,g,b])
    cv2.imshow('Cur',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # render the image
    #output.Render(img)

    # update the title bar
    #output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
