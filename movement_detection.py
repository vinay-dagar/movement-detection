import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
ret, frame1 = capture.read()
ret, frame2 = capture.read()

while capture.isOpened():
    diff = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(threshold, None, iterations=3)
    controus, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.drawContours(frame1, controus, -1, (23, 234, 87), 2)

    cv.imshow('Movement Detection', frame1)
    frame1 = frame2
    ret, frame2 = capture.read()

    if cv.waitKey(40) == 27:
        break

capture.release()
cv.destroyAllWindows()