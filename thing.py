import cv2 as cv
import numpy as np

def nothing(x):
    pass
def set_brightness(val):
    global brightness
    brightness = val

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
detect_faces = False

cv.namedWindow('Filter Window')


cv.createTrackbar('Grayscale', 'Filter Window', 0, 1, nothing)
cv.createTrackbar('HSV Channel', 'Filter Window', 0, 4, nothing)
cv.createTrackbar('YUV Channel', 'Filter Window', 0, 4, nothing)
cv.createTrackbar('Blur', 'Filter Window', 0, 10, nothing)
cv.createTrackbar('Canny Edge', 'Filter Window', 0, 100, nothing)
cv.createTrackbar("Brightness", "Filter Window", 0, 100, set_brightness)

x = 640
y = 480
xi = 0
yi = 0
rot = 0
angle = 0
current_morph = None
camera = cv.VideoCapture(0) 
brightness = 50
mirror = False
if not camera.isOpened():
    raise IOError("Camera not accessed.")

while True:
    grayscale = cv.getTrackbarPos('Grayscale', 'Filter Window')
    channel = cv.getTrackbarPos('HSV Channel', 'Filter Window')
    yuv_channel = cv.getTrackbarPos('YUV Channel', 'Filter Window')
    blur = cv.getTrackbarPos('Blur', 'Filter Window')
    edge = cv.getTrackbarPos('Canny Edge', 'Filter Window')
    kernel = None
    
    ret, frame = camera.read()
    col,row = frame.shape[:2]
    key = cv.waitKey(1)
    frame = cv.flip(frame,1)

    trans = np.float32([[1,0,xi],[0,1,yi]])
    rotate_mat = cv.getRotationMatrix2D((col/2,row/2),angle,1)
    alpha = brightness / 50.0

    frame = cv.warpAffine(frame,rotate_mat,(row,col))
    frame = cv.resize(frame, (x, y))
    frame = cv.convertScaleAbs(frame, alpha=alpha,beta=0)
    frame = cv.warpAffine(frame, trans, (640, 480))
    frame = cv.morphologyEx(frame,current_morph,kernel)
###############################################################
    if key == ord('f'):
        mirror = not mirror


    if mirror:
        frame = cv.flip(frame, 1)
    if key == ord("'"):
        x = 640
        y = 480
        xi = 0
        yi = 0
        rot = 0
        angle = 0
        current_morph = None
        frame = cv.morphologyEx(frame,current_morph,kernel)
        detect_faces = False
        frame = cv.flip(frame, 0)
################################################################
    if detect_faces:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (a,b,c,d ) in faces:
                cv.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 2)
################################################################
    if key == ord('k'):
        detect_faces = not detect_faces
################################################################
    if key == ord('o'):
        x += 10
        y += 10
    elif key == ord('p'):
        x -=10
        y -= 10
    elif key == ord('d'):
        xi += 10
    elif key == ord('a'):
        xi -= 10
    elif key == ord('w'):
        yi += 10
    elif key == ord('s'):
        yi -= 10
################################################################
    elif key == ord("]"):
            angle += 10
            angle %= 360
################################################################
    elif key == ord("["):
            angle += -10
            angle %= 360
################################################################
    # Apply filters based on trackbar settings
    if grayscale > 0:
        channel = 0
        yuv_channel = 0
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
################################################################
    if channel != 0:
        yuv_channel = 0
        grayscale = 0
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        if channel == 1:
            frame = hsv_frame[:, :, 0]  

        elif channel == 2:
            frame = hsv_frame[:, :, 1]  

        elif channel == 3:
            frame = hsv_frame[:, :, 2]

        elif channel == 4:
            frame = hsv_frame
################################################################
    if yuv_channel != 0:
        grayscale = 0
        channel = 0
        yuv_frame = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
        if yuv_channel == 1:
            frame = yuv_frame[:, :, 0]  

        elif yuv_channel == 2:
            frame = yuv_frame[:, :, 1]  

        elif yuv_channel == 3:
            frame = yuv_frame[:, :, 2]

        elif yuv_channel == 4:
            frame = yuv_frame
################################################################
    if blur > 0:
        frame = cv.GaussianBlur(frame, (2 * blur + 1, 2 * blur + 1), 0)
################################################################
    if edge > 0:
        frame = cv.Canny(frame, edge,edge*2)
################################################################
    if key == ord('0'):
        cv.destroyAllWindows()
        filename = input("Enter name: ")
        cv.imwrite(f"{filename}.png", frame)
        break
################################################################
    elif key == ord('1'):
        current_morph = cv.MORPH_GRADIENT
        kernel = np.ones((5,5), np.uint8)
################################################################
    elif key == ord('2'):
        current_morph = cv.MORPH_BLACKHAT
        kernel = np.ones((5,5), np.uint8)
################################################################
    elif key == ord('3'):
        current_morph = cv.MORPH_CLOSE
        kernel = np.ones((5,5), np.uint8)
################################################################
    elif key == ord('4'):
        current_morph = cv.MORPH_CROSS
        kernel = np.ones((5,5), np.uint8)
################################################################
    elif key == ord('5'):
        current_morph = cv.MORPH_ERODE
        kernel = np.ones((5,5), np.uint8)
################################################################
    if key == ord('0'):
        cv.destroyAllWindows()
        filename = input("Enter name: ")
        cv.imwrite(f"{filename}.png", frame)
        print(f'Successfully saved {filename}.png')
        break
################################################################
    if key == 27:
        cv.destroyAllWindows()
        break
################################################################
    cv.imshow("Image", frame)
################################################################
camera.release()
cv.destroyAllWindows()
