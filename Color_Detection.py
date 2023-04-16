import cv2

cap = cv2.VideoCapture(0)

while True:
    retrn, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # color range for color detection. (eg. blue color)
    lo_clr_range = (90, 50, 50)
    up_clr_range = (128, 255, 255)

    mask = cv2.inRange(hsv, lo_clr_range, up_clr_range)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    output = cv2.imshow('Main Scan', frame)
    output = cv2.imshow('Color Range Mask', mask)
    output = cv2.imshow('Detected Color', result)

    if cv2.waitKey(1) == 27:  # replace [ord('q')] with 27 for Esc. key
        break

cap.release()

cv2.destroyAllWindows()

# inRange
# bitwise_and
# lo_clr_range = (90, 50, 50)
# up_clr_range = (128, 255, 255)
