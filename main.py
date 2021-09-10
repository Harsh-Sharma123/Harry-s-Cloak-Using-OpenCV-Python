import cv2
import numpy as np

def hello(x):
    print("")

cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("Bars")

cv2.createTrackbar("Upper Hue","Bars", 171, 194, hello)
cv2.createTrackbar("Lower Hue","Bars", 68, 100, hello)
cv2.createTrackbar("Upper Saturation","Bars", 255, 255, hello)
cv2.createTrackbar("Lower Saturation","Bars", 55, 255, hello)
cv2.createTrackbar("Upper Value","Bars", 255, 255, hello)
cv2.createTrackbar("Lower Value","Bars", 54, 255, hello)


# Capturing the initial frame for creation of background
while(True):
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    # check if we get frame then break
    if(ret):
        break

# Start capturing the frame for the actual magic
while(True):
    ret, frame = cap.read()
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # getting the HSV values for the masking of cloak
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Bars")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Bars")
    upper_value = cv2.getTrackbarPos("Upper Value", "Bars")
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Bars")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Bars")
    lower_value = cv2.getTrackbarPos("Lower Value", "Bars")

    # kernel to be used for dilation
    kernel = np.ones((3,3), np.uint8)

    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])

    mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
    mask  = cv2.medianBlur(mask, 3)
    mask_inv = 255-mask
    mask = cv2.dilate(mask, kernel, 5)

    # The mixing of frames in a combination to achieve the required frame
    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]
    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b,g,r))


    b = init_frame[:,:,0]
    g = init_frame[:,:,0]
    r = init_frame[:,:,0]
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    blanket_area = cv2.merge((b, g, r))

    final = cv2.bitwise_or(frame_inv, blanket_area)

    cv2.imshow("Harry's Cloak", final)
    cv2.imshow("Original", frame)

    # cv2.imshow("Mask", mask)
    # cv2.imshow('Inv', mask_inv)

    # cv2.imshow('FrameInv', frame_inv)
    # cv2.imshow("Blank", blanket_area)

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()