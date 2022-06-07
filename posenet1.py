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
    with open('/home/jetsontx23/jetson-inference/python/examples/pose_pre_file.txt', 'w+') as file_write:
        file_write.write(str(pose.Keypoints))
    file_write.close()

    with open('/home/jetsontx23/jetson-inference/python/examples/pose_pre_file.txt', 'r+') as file_read:
        while True:
            lines = file_read.readline()
            #print(lines)
            if ("left_shoulder") in lines:
                lines = next(file_read)
                pre_left_shoulder_x = re.findall(r"\d+\.?\d*",lines)
                pre_left_shoulder_x = float(pre_left_shoulder_x[0])
                #print("left_shoulder_x",left_shoulder_x)
    
                lines_y = next(file_read)
                pre_left_shoulder_y = re.findall(r"\d+\.?\d*",lines_y)
                pre_left_shoulder_y = float(pre_left_shoulder_y[0])
                #print("left_shoulder_y",left_shoulder_y)

            if ("right_shoulder") in lines:
                lines = next(file_read)
                pre_right_shoulder_x = re.findall(r"\d+\.?\d*",lines)
                pre_right_shoulder_x = float(pre_right_shoulder_x[0])
                #print("right_shoulder_x",right_shoulder_x)
    
                lines_y = next(file_read)
                pre_right_shoulder_y = re.findall(r"\d+\.?\d*",lines_y)
                pre_right_shoulder_y = float(pre_right_shoulder_y[0])
                #print("right_shoulder_y",right_shoulder_y)

            if ("right_knee") in lines:
                lines = next(file_read)
                pre_right_knee_x = re.findall(r"\d+\.?\d*",lines)
                pre_right_knee_x = float(pre_right_knee_x[0])
                #print("right_shoulder_x",right_shoulder_x)
    
                lines_y = next(file_read)
                pre_right_knee_y = re.findall(r"\d+\.?\d*",lines_y)
                pre_right_knee_y = float(pre_right_knee_y[0])
                #print("right_shoulder_y",right_shoulder_y)

            if ("right_ankle") in lines:
                lines = next(file_read)
                pre_right_ankle_x = re.findall(r"\d+\.?\d*",lines)
                pre_right_ankle_x = float(pre_right_ankle_x[0])
                #print("right_shoulder_x",right_shoulder_x)
    
                lines_y = next(file_read)
                pre_right_ankle_y = re.findall(r"\d+\.?\d*",lines_y)
                pre_right_ankle_y = float(pre_right_ankle_y[0])
                #print("right_shoulder_y",right_shoulder_y)

            if ("]") in lines:
                break
   
pre_dy_leg = abs(pre_right_knee_y - pre_right_ankle_y)
pre_dx_leg = abs(pre_right_knee_x - pre_right_ankle_x)
pre_dz_2_leg = pre_dy_leg ** 2 + pre_dx_leg ** 2
pre_dz_leg = math.sqrt(pre_dz_2_leg) 
pre_angle1_leg = math.acos(pre_dx_leg / pre_dz_leg)
pre_angle1_leg = pre_angle1_leg * 180 / math.pi
#print("pre_action_leg",angle1_leg)



pre_dy_shoulder = abs(pre_right_shoulder_y - pre_left_shoulder_y)
pre_dx_shoulder = abs(pre_right_shoulder_x - pre_left_shoulder_x)
pre_dz_2_shoulder = pre_dy_shoulder ** 2 + pre_dx_shoulder ** 2
pre_dz_shoulder = math.sqrt(pre_dz_2_shoulder) 
pre_angle1_shoulder = math.acos(pre_dx_shoulder / pre_dz_shoulder)
pre_angle1_shoulder = pre_angle1_shoulder * 180 / math.pi
#print("pre_action_shoulder",pre_angle1_shoulder)






# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
#input = jetson.utils.videoSource(opt.input_URI, argv=["--input-flip=route-180"])
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

flag_left_shoulder = 0
flag_right_shoulder = 0
flag_right_knee = 0
flag_right_ankle = 0

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)
    #img = cv2.VideoCapture("/dev/video2")
    # print the pose results
    # print("detected {:d} objects in image".format(len(poses)))

    img=numpy.array(img)

    for pose in poses:
        #print(pose)
        #print(pose.Keypoints)
        #print('Links', pose.Links)

        with open('/home/jetsontx23/jetson-inference/python/examples/pose1_file.txt', 'w+') as file0_write:
            file0_write.write(str(pose))
        file0_write.close()
	# write file

        with open('/home/jetsontx23/jetson-inference/python/examples/pose1_file.txt', 'r+') as file3_read:
            while True:
                lines = file3_read.readline()
                #print(lines)
                if ("<poseNet.ObjectPose") in lines:
                    lines = next(file3_read)
                    cur_id = re.findall(r"\d+\.?\d*",lines)
                    cur_id = float(cur_id[0])
                    break


        if (cur_id == 0):

            with open('/home/jetsontx23/jetson-inference/python/examples/pose_file.txt', 'w+') as file1_write:
                #file1_write.write(str(pose))
                file1_write.write(str(pose.Keypoints))
            file1_write.close()

        with open('/home/jetsontx23/jetson-inference/python/examples/pose_file.txt', 'r+') as file2_write:
            while True:
                lines = file2_write.readline()
                #print(lines)

                if ("left_shoulder") in lines:
                    flag_left_shoulder = 1
                    lines = next(file2_write)
                    left_shoulder_x = re.findall(r"\d+\.?\d*",lines)
                    left_shoulder_x = float(left_shoulder_x[0])
                    #print(left_shoulder_x)

                    lines_y = next(file2_write)
                    left_shoulder_y = re.findall(r"\d+\.?\d*",lines_y)
                    left_shoulder_y = float(left_shoulder_y[0])
                    #print(left_shoulder_y)

                if ("right_shoulder") in lines:
                    flag_right_shoulder = 1
                    lines = next(file2_write)
                    right_shoulder_x = re.findall(r"\d+\.?\d*",lines)
                    right_shoulder_x = float(right_shoulder_x[0])
                    #print(right_shoulder_x)

                    lines_y = next(file2_write)
                    right_shoulder_y = re.findall(r"\d+\.?\d*",lines_y)
                    right_shoulder_y = float(right_shoulder_y[0])
                    #print(right_shoulder_y)


                if ("right_knee") in lines:
                    flag_right_knee = 1
                    lines = next(file2_write)
                    right_knee_x = re.findall(r"\d+\.?\d*",lines)
                    right_knee_x = float(right_knee_x[0])
                    #print("right_shoulder_x",right_shoulder_x)
    
                    lines_y = next(file2_write)
                    right_knee_y = re.findall(r"\d+\.?\d*",lines_y)
                    right_knee_y = float(right_knee_y[0])
                    #print("right_shoulder_y",right_shoulder_y)


                if ("right_ankle") in lines:
                    flag_right_ankle = 1
                    lines = next(file2_write)
                    right_ankle_x = re.findall(r"\d+\.?\d*",lines)
                    right_ankle_x = float(right_ankle_x[0])
                    #print("right_shoulder_x",right_shoulder_x)
    
                    lines_y = next(file2_write)
                    right_ankle_y = re.findall(r"\d+\.?\d*",lines_y)
                    right_ankle_y = float(right_ankle_y[0])
                    #print("right_shoulder_y",right_shoulder_y)


                if ("]") in lines:
                    break

    #print(flag_left_shoulder,flag_right_shoulder)

        if (flag_left_shoulder == 1 & flag_right_shoulder == 1):
            dy_shoulder = abs(right_shoulder_y - left_shoulder_y)
            dx_shoulder = abs(right_shoulder_x - left_shoulder_x)
            dz_2_shoulder = dy_shoulder ** 2 + dx_shoulder ** 2
            dz_shoulder = math.sqrt(dz_2_shoulder) 
            angle_shoulder = math.acos(dx_shoulder / dz_shoulder)
            angle_shoulder = angle_shoulder * 180 / math.pi
            print("cur_action_shoulder",angle_shoulder)

            if abs(angle_shoulder - pre_angle1_shoulder) <= 5:
                a="shoulders level"
            else:
                a="shoulders need to be level"
            print(a)

            cv2.putText(img,a,(50,400),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),3,cv2.LINE_AA)

        if (flag_right_ankle == 1 & flag_right_knee == 1):
            dy_leg = abs(right_knee_y - right_ankle_y)
            dx_leg = abs(right_knee_x - right_ankle_x)
            dz_2_leg = dy_leg ** 2 + dx_leg ** 2
            dz_leg = math.sqrt(dz_2_leg) 
            angle_leg = math.acos(dx_leg / dz_leg)
            angle_leg = angle_leg * 180 / math.pi
            print("cur_action_leg",angle_leg)

            if abs(angle_leg - pre_angle1_leg) <= 5:
                b="right leg vertical"
            else:
                b="right leg not vertical"
            print(b)

            cv2.putText(img,b,(50,500),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),3,cv2.LINE_AA)

    

    
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
