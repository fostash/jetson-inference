#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import time
import os

from paho.mqtt.client import Client

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=jetson.inference.imageNet.Usage() +
                                        jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0",
                    help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)
detectnet = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
videoSource = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
# output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv + is_headless)
# font = jetson.utils.cudaFont()
client = Client(client_id="client_1")

print("connect to mqtt")
client.connect("localhost")

# process frames until the user exits
previous_400 = int(round(time.time() * 1000))
previous_2400 = previous_400
captured_image_path = os.environ.get('CAPTURED_IMAGES_PATH')
check_still_image_path = os.environ.get('CHECK_STILL_IMAGES_PATH')

while True:
    actual_400 = int(round(time.time() * 1000))
    actual_2400 = actual_400
    if (actual_400 - previous_400) > 400:
        # capture the next image
        img = videoSource.Capture()

        # classify the image
        class_id, confidence = net.Classify(img)

        # find the object description
        class_desc = net.GetClassDesc(class_id)

        json_value = '{ "path": "%s", "name": "%s", "timestamp": %d, "format": "jpeg", "class_id": %d, ' \
                     '"class_desc": "%s", "confidence": %d }' \
                     % (captured_image_path, actual_400, actual_400, class_id, class_desc, confidence)
        if confidence < 0.4:
            print (json_value)
            # push message to mqtt as classification unknown
            client.publish(topic="images/captured/unknown_classification", payload=json_value)
        elif 0.4 < confidence < 0.5:
            # push message to mqtt as maybe known
            client.publish(topic="images/captured/partially/maybe_known", payload=json_value)
        elif 0.6 < confidence < 0.8:
            # push message to mqtt as quite sure classification
            client.publish(topic="images/captures/almost_sure", payload=json_value)
        elif confidence > 0.8:
            # push message to mqtt the deletion of the image
            client.publish(topic="images/captured/recognized", payload=json_value)

        output_captured = jetson.utils.videoOutput("file://{}/{}.jpg".format(captured_image_path, actual_400),
                                                   argv=sys.argv + is_headless)
        output_captured.Render(img)
        previous_400 = actual_400

        if (actual_2400 - previous_2400) > 2400:
            output_still = jetson.utils.videoOutput("file://{}/{}.jpg".format(check_still_image_path, actual_2400),
                                                    argv=sys.argv + is_headless)
            output_still.Render(img)
            previous_2400 = actual_2400

        # update the title bar
        # output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        # print out performance info
        # net.PrintProfilerTimes()

        # detect objects in the image (with overlay)
        detections = detectnet.Detect(img, overlay=opt.overlay)

        # print the detections
        print("detected {:d} objects in image".format(len(detections)))
        for detection in detections:
            print(detection)

        # exit on input/output EOS
        if not videoSource.IsStreaming():  # or not output.IsStreaming():
            client.disconnect()
            break
