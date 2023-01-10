import cv2
import numpy as np


# creating trackbars
def detect(x):
    print(x)


# creating trackbars

cv2.namedWindow('Trackbars')
cv2.createTrackbar('Hue_min', "Trackbars", 0, 179, detect)
cv2.createTrackbar('Hue_max', "Trackbars", 0, 179, detect)
cv2.createTrackbar('Sat_min', "Trackbars", 0, 254, detect)
cv2.createTrackbar('Sat_max', "Trackbars", 0, 254, detect)
cv2.createTrackbar('Val_min', "Trackbars", 0, 254, detect)
cv2.createTrackbar('Val_max', "Trackbars", 0, 254, detect)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

# get frame of camera and resize it

# def get_frame(cap, scaling_factor):
    #
    # frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # return frame


# main loop

cap = cv2.VideoCapture(0)
scaling_factor = 0.8
while True:
    ret, frame = cap.read()
    # frame = get_frame(cap, scaling_factor)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos('Hue_min', 'Trackbars')
    hue_max = cv2.getTrackbarPos('Hue_max', 'Trackbars')
    sat_min = cv2.getTrackbarPos('Sat_min', 'Trackbars')
    sat_max = cv2.getTrackbarPos('Sat_max', 'Trackbars')
    value_min = cv2.getTrackbarPos('Val_min', 'Trackbars')
    value_max = cv2.getTrackbarPos('Val_max', 'Trackbars')

    # define threshold
    lower_hsv = np.array([hue_min, sat_min, value_min])
    upper_hsv = np.array([hue_max, sat_max, value_max])

    # create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cv2.imshow('Mask', mask)
    # apply mask
    img = cv2.bitwise_xor(frame, frame, mask=mask)
    cv2.imshow('Original', frame)
    cv2.imshow('Output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
