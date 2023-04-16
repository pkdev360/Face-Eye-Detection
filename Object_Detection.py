import cv2

img = cv2.resize(cv2.imread('assets/soccer_practice.jpg'),
                 (0, 0), fx=0.6, fy=0.6)
templet = cv2.resize(cv2.imread('assets/ball.png', 0), (0, 0), fx=0.6, fy=0.6)
h, w = templet.shape

methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]

# Object (soccer ball) Detection using each method.
for method in methods:
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img2 = img.copy()
    match = cv2.matchTemplate(gray, templet, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + h, top_left[1] + w)

    cv2.rectangle(img2, top_left, bottom_right, (0, 0, 255), 5)

    cv2.imshow(str(method), img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# matchTemplate
# minMaxLoc
# rectangle
