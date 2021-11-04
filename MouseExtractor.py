import cv2

import numpy as np



class MouseExtractor:
    
    def __init__(self, filterKernelSize=5, backSubMOG2=True, blobAreaThreshold=500):
        '''
        Parameters:
            filterKernelSize (int): size of median filter
            backSubMOG2 (bool): Decides which algorithm to use for background subtraction.
                                True -> use MOG2
                                False -> use KNN
            blobAreaThreshold (int): minimum area of detected contour to consider a mouse.
        '''
        
        # Kernel size for median filter
        self.filterKernelSize = 5
        
        # Initialize bounding box variables as 0.
        self.bb_x, self.bb_y, self.bb_w, self.bb_h = (0,) * 4
        
        # Initialize background subtraction algorithm.
        if backSubMOG2:
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            self.backSub = cv2.createBackgroundSubtractorKNN()
            
        # Minimum blob area. Default was found via data exploration.
        self.blobAreaThreshold = blobAreaThreshold
        
    
    
    
    def preprocess(self, frame):
        '''
        Run median blur, background subtraction, and binary thresholding on frame.
        '''
        
        # Filter the image before running background subtraction algorithm
        
        #frame = cv2.GaussianBlur(frame, (self.filterKernelSize,self.filterKernelSize), cv2.BORDER_DEFAULT)
        frame = cv2.medianBlur(frame, self.filterKernelSize)
        #frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
        # Run background subtraction algorithm. Keep only the foreground/shadow mask.
        frame = self.backSub.apply(frame)
        
        # Re-filter the background-subtracted image.
        frame = cv2.medianBlur(frame, self.filterKernelSize)
        
        # Remove the gray shadow contours, reducing image to only black or white.
        frame = cv2.inRange(frame, 254, 255)
        
        return frame
    
    
    
    def getPosition(self, frame):
        '''
        Extract the mouse's position from the frame.
        
        Returns: x: horizontal midpoint of the mouse's latest bounding box
                 y: vertical midpoint   of the mouse's latest bounding box
                 r_w: horizontal radius of the mouse's latest bounding box (equals width / 2)
                 r_h: vertical radius   of the mouse's latest bounding box (equals height / 2)
                 binary: the thresholded binary frame. Useful for visualizing what the algorithm is producing.
        '''
        
        frame = self.preprocess(frame)
        
          
        # Find all contours in the thresholded frame
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # If any contour was found:
        if len(contours) != 0:
            contours = np.array(contours)
            
            # Find largest contour (which should be mouse, but is not always)         
            largest_contour = max(contours, key = cv2.contourArea)
            a = cv2.contourArea(largest_contour)
            
            # Ignore small contours (which are likely just illumination changes).
            # If the largest contour is big enough:
            if a > self.blobAreaThreshold:
                # Update the dimensions of the bounding box.
                self.bb_x, self.bb_y, self.bb_w, self.bb_h = cv2.boundingRect(largest_contour)
            
        
        # Calculate horizontal and vertical radii of bounding box.
        r_w = self.bb_w // 2
        r_h = self.bb_h // 2
            
        # Return midpoint and radii of most recent bounding box.
        # Also return the thresholded binary frame.
        return self.bb_x + r_w, self.bb_y + r_h, r_w, r_h, frame
        