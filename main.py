import os
import cv2
import numpy as np
import importlib.util
from threading import Thread
import RPi.GPIO as GPIO
import time

from image import *
from pump import *

MODEL_NAME = 'TFLite_model'
PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, 'detect.tflite')
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_NAME, 'labelmap.txt')
min_conf_threshold = 0.5

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
resW=1280
resH=720

floating_model = (input_details[0]['dtype'] == np.float32)

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

input_mean = 127.5
input_std = 127.5

def pet_detector(frame, detection_time, cooldown):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    #boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            obj_name = labels[int(classes[i])]
            if obj_name == 'cat' or obj_name == 'teddy bear':
                detection_time += 1
            else:
                detection_time = 0
                no_cat()
                if (time.process_time() - cooldown > 5):
                    switch_pump(False)

            print('Looking at a cat for %s frames' % detection_time)

            if detection_time >= 2:
                cooldown = time.process_time()
                cat()
                switch_pump(True)

    return frame, detection_time

videostream = VideoStream(resolution=(resW, resH), framerate=30).start()
time.sleep(1)

detection_time = 0
cooldown = 0
while(True):
    frame = videostream.read()
    frame, detection_time = pet_detector(frame, detection_time, cooldown)
    cv2.imshow('Cat-detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
