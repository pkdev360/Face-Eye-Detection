import cv2

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_csscade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')


while True:
    rtrn, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

        eye_roi_gray = gray[y:y+h, x:x+w]
        eye_roi_color = frame[y:y+h, x:x+w]
        eyes = eye_csscade.detectMultiScale(eye_roi_gray, 1.2, 5)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(eye_roi_color, (xe, ye),
                          (xe+we, ye+he), (0, 0, 255), 5)

    cv2.imshow('Output', frame)

    if cv2.waitKey(1) == ord('q'):  # replace ord('q') with 27 for Esc. key
        break

cam.release()
cv2.destroyAllWindows()


# cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
# detectMultiScale
