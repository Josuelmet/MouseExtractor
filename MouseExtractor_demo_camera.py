# Source: https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/ 

import cv2
import depthai


import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import time
from datetime import datetime

from MouseExtractor import MouseExtractor



PI = True


DISPLAY = True # True = show live stream, False = don't show live stream.

RECORD = True # True = record stream, False = don't record stream.
frameSize = (300, 300) # 300x300 for ease of use with Pi 
fileName = 'output.avi'


# Instantiate position extractor
mouse = MouseExtractor()

# Initialize position lists
positions_x = deque()
positions_y = deque()
timestamps = deque()
START_TIME = None




# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(frameSize[0], frameSize[1])
cam_rgb.setInterleaved(False) # Optional
if PI:
    fps = 5
    cam_rgb.setFps(fps)
else:
    fps = 240
    # Let the colorCamera have its default FPS.



# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)



if RECORD:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename=fileName, fourcc=fourcc, apiPreference=0,
                          fps=fps, frameSize=frameSize)
    
    

if DISPLAY:
    print('Press q to exit.')
else:
    print('Press CTRL + c to exit.')


# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline, usb2Mode=False) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    #frame = None


    # Main host-side application loop
    try:
        while True:
            
            frame = None
        
            # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
            in_rgb = q_rgb.tryGet()

            if in_rgb is not None:
                # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                frame = in_rgb.getCvFrame()

            if frame is not None:
                
                # Get bounding box information about mouse
                (x, y, r_w, r_h, binary) = mouse.getPosition(frame)
                
                positions_x.append(x)
                positions_y.append(y)
                
                if START_TIME is None:
                    START_TIME = time.time()
                    timestamps.append(0)
                else:
                    timestamps.append(time.time() - START_TIME)
                    
                
                # Draw bounding box on frame
                if DISPLAY or RECORD:
                    cv2.rectangle(frame, (x-r_w, y-r_h), (x+r_w, y+r_h), (255, 255, 0), 5)
                    
                    cv2.putText(frame, str(datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,255), 1, cv2.LINE_AA)
                    
                    
                    
                # If you want to display live video feed:
                if DISPLAY:            
                    # After all the drawing is finished, we show the frame on the screen
                    cv2.imshow("Tracking preview", frame)
                    
                    cv2.imshow('Binary frame', binary)
                
                if RECORD:
                    out.write(frame)

            # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == 0x03:
                break
            
    except KeyboardInterrupt:
        print('terminated')
        pass
        
        
cv2.destroyAllWindows()

# Plot position vs. time
plt.figure()
plt.title('Position vs. Time')
plt.plot(timestamps, positions_x, label='x')
plt.plot(timestamps, positions_y, label='y')
plt.xlabel('Time (s)')
plt.legend()
plt.show()


if RECORD:
    out.release()