import math
from collections import Counter
import cv2
# import dlib
import numpy as np
from inpaint_utilities import *
from pyheal import *

frame_counter=0

kernel = np.ones((3,3), np.uint8)


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def distance(point1, point2):
    dist = math.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)
    return dist


def return_color_mask(min_color_value, max_color_value, image_hsv, morph_kernel, original_image):

    color_mask = cv2.inRange(image_hsv, min_color_value, max_color_value)
    resultant_color_image = cv2.bitwise_and(image_hsv, image_hsv, mask=color_mask)

    h, s, resultant_color_gray = cv2.split(resultant_color_image)

    _, thresh = cv2.threshold(resultant_color_gray, 0, 255, 0)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
    thresh = cv2.dilate(thresh, morph_kernel, iterations=3)
    
    ref_image = cv2.imwrite('ref_image.jpg',thresh)
    color_image = cv2.bitwise_or(original_image, ref_image, mask=thresh)
    
    return color_image,thresh


check_list = []
most_pixel = []
cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    copied_image = image.copy()
    copied_image1= image.copy()
    imgray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_min = np.array([30, 35, 35], np.uint8)
    green_max = np.array([90, 240, 240], np.uint8)
   

    green_mask, thresh = return_color_mask(green_min, green_max, hsv_image, kernel, image)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        mask = np.zeros_like(image)
        mask = cv2.bitwise_not(mask)
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        mask = cv2.bitwise_not(mask)
        cv2.drawContours(mask, [cnt], -1, (0, 0, 0), -1)
        mask = cv2.bitwise_not(mask)
        M = cv2.moments(cnt)
        dst = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
        magic_image = cv2.bitwise_and(dst, dst, mask=mask)
        edges = cv2.Canny(magic_image, 50, 300, apertureSize=3)

        dst = cv2.inpaint(dst, edges, 1, cv2.INPAINT_TELEA)
        magic_image = cv2.bitwise_and(dst, dst, mask=mask)
        edges = cv2.Canny(magic_image, 50, 300, apertureSize=3)
        dst = cv2.inpaint(dst, edges, 1, cv2.INPAINT_TELEA)
        cv2.imshow('result_frame', dst)

    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
