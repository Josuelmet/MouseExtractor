import cv2

import numpy as np
from matplotlib import pyplot as plt

from MouseExtractor import MouseExtractor



input_video_path = 'temp_mouse.mp4'
FPS = 1000


cap = cv2.VideoCapture(input_video_path)

# Instantiate position extractor
mouse = MouseExtractor()


# Initialize position lists
positions_x = []
positions_y = []    



while(cap.isOpened()):
    
    # Read a new frame
    ok, frame = cap.read()
    frame = cv2.UMat(frame) # Convert frame to UMat to use GPU via OpenCL library.
    
    if ok and frame is not None:
        
        (x, y, r_w, r_h, binary) = mouse.getPosition(frame)
        
        positions_x = np.append(positions_x, x)
        positions_y = np.append(positions_y, y)
        
        
        cv2.rectangle(frame, (x-r_w, y-r_h), (x+r_w, y+r_h), (255, 255, 0), 5)
            
        # Display result        
        cv2.imshow("Tracking", frame)
        
        cv2.imshow('Binary frame', binary)

        if cv2.waitKey(1000 // FPS) == ord('q'):
            break
        
        
    else:
        break


cap.release()
cv2.destroyAllWindows()


plt.figure()
plt.title('Position vs. Time')
plt.plot(positions_x, label='x')
plt.plot(positions_y, label='y')
plt.legend()
plt.show()