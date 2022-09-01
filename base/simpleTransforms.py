import os
import cv2
import random
import numpy as np
import SimpleITK as sitk

import torch
from PIL import Image

# simple processing
class makeRGB(object):
    """
    make a grayscale image RGB
    """

    def __init__(self):
        self.initalized = True

    def __call__(self, img):
        return img.convert('RGB')

### Cardiac project specific processing ###
class AddCardiacRespiratoryLines(object):
    """Add a respiratory line to the given image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        input_dim (tuple): size of the image
        base_path (string): path to the line images 
    """
    def __init__(self, base_path, p=0.5):
        self.p = p
        self.base_path = base_path

    def __call__(self, img):
    
        if torch.rand(1) < self.p:  # if randomly selected to add a respiratory line
            resp_line = random.choice(['resp_3.png', 'resp_4.png', 'resp_5.png', 'resp_6.png', 'resp_7.png'])
            img_line = Image.open(os.path.join(self.base_path, resp_line)).convert('RGB')
            img_line = img_line.resize(img.size)
            img_line_np = np.array(img_line)
            
            img = np.array(img)
            img[img_line_np != 0] = 135
            
            img = Image.fromarray(img)

        return img

class SegmentCardiacEcho(object):
    """
    code to segment just the ultrasound image of the cardiac ultrasound frames
    Args:
        frame (np.array): the raw ultrasound array frame
        simpleCrop (bool): option for simply cropping a standardized dataset input or using a more complex, connected components method
    output:
        frame_crop (np.array): segmented cardiac ultrasound
    """
    def __init__(self, simpleCrop=False):
        self.simpleCrop = simpleCrop

    def __call__(self, frame):
        if self.simpleCrop:  # if it's a standarized ultrasound, just crop a rectangle
            frame_h = int(frame.shape[0] * 0.1)
            frame_w = int(frame.shape[1] * 0.25)
            frame_crop = frame[frame_h:-frame_h, frame_w:-frame_w]
        else:  # if not standardized, use James' code
            ret2, threshhold = cv2.threshold(frame, 29, 255, 0)
            contours, hierarchy = cv2.findContours(threshhold, 1, 2)
            # Approx contour
            cnt = contours[0]
            largest = cv2.contourArea(cnt)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            cnt = contours[0]
            # Central points and area
            moments = cv2.moments(cnt)
            cent_x = int(moments['m10'] / moments['m00'])
            cent_y = int(moments['m01'] / moments['m00'])
            shape_area = cv2.contourArea(cnt)
            shape_perim = cv2.arcLength(cnt, True)
            epsilon = 0.01 * shape_perim
            approximation = cv2.approxPolyDP(cnt, epsilon, True)
            convex_hull = cv2.convexHull(cnt)
            contour_mask = np.zeros(frame.shape, np.uint8)
            contour_mask = cv2.drawContours(contour_mask, [convex_hull], 0, 255, -1)

            frame_crop = frame * contour_mask

        return frame_crop

class AblateCardiacEcho(object):
    """
    code to remove different quadrants of the cardiac ultrasound frames
    input:
        frame_crop: the raw ultrasound array to crop a quadrant from
        quadrant: option to remove a quadrant of the image for ablation study
    output:
        frame_crop: segmented cardiac ultrasound (w/w0 quadrant ablation)
    """
    def __init__(self, quadrant=0):
        self.quadrant = quadrant

    def __call__(self, frame_center):
        frame_crop = frame_center.copy()

        frame_center_h = int(frame_crop.shape[0] * 0.6)
        frame_center_w = int(frame_crop.shape[1] * 0.6)
        if quadrant in [1, 2, 3, 4]:  # removing a quadrant
            if quadrant == 1:
                frame_crop[:frame_center_h, :frame_center_w] = 0
            elif quadrant == 2:
                frame_crop[frame_center_h:, :frame_center_w] = 0
            elif quadrant == 3:
                frame_crop[:frame_center_h, frame_center_w:] = 0
            else:
                frame_crop[frame_center_h:, frame_center_w:] = 0
        elif quadrant in [5, 6, 7, 8]:  # keeping only the quadrant
            zeros = np.zeros(frame_crop.shape)
            if quadrant == 5:
                zeros[:frame_center_h, :frame_center_w] = 1
            elif quadrant == 6:
                zeros[frame_center_h:, :frame_center_w] = 1
            elif quadrant == 7:
                zeros[:frame_center_h, frame_center_w:] = 1
            else:
                zeros[frame_center_h:, frame_center_w:] = 1
            frame_crop = frame_crop * zeros
        else:  # this is removing or keeping the cross only
            frame_center_h_buffer = int(frame_crop.shape[0] * 0.05)
            frame_center_w_buffer = int(frame_crop.shape[1] * 0.05)
            if quadrant == 'cross_only':  # only keeping the cross
                ones = np.ones(frame_crop.shape)
                ones[:frame_center_h-frame_center_h_buffer, :frame_center_w-frame_center_w_buffer] = 0
                ones[frame_center_h+frame_center_h_buffer:, :frame_center_w-frame_center_w_buffer] = 0
                ones[:frame_center_h-frame_center_h_buffer, frame_center_w+frame_center_w_buffer:] = 0
                ones[frame_center_h+frame_center_h_buffer:, frame_center_w+frame_center_w_buffer:] = 0
                frame_crop = frame_crop * ones
            else:  # 'cross_remove' removing the cross
                zeros = np.zeros(frame_crop.shape)
                zeros[:frame_center_h-frame_center_h_buffer, :frame_center_w-frame_center_w_buffer] = 1
                zeros[frame_center_h+frame_center_h_buffer:, :frame_center_w-frame_center_w_buffer] = 1
                zeros[:frame_center_h-frame_center_h_buffer, frame_center_w+frame_center_w_buffer:] = 1
                zeros[frame_center_h+frame_center_h_buffer:, frame_center_w+frame_center_w_buffer:] = 1
                frame_crop = frame_crop * zeros
                    
        return frame_crop