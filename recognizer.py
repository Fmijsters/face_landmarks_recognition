
import cv2
import numpy as np
import os
import dlib
import sys
import time
# from imutils import face_utils
subjects  =["","Fabian Mijsters","Random Person"]

global face_cascade, predictor,clahe,detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()


def instantiateWebCam(source):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print("Succesfully opened the camera!")
    else:
        print("Failed to open the camera try to change the source of the camera")
        return None
    return cap

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords

cap =  instantiateWebCam(0)
def align(image,shape): 
    cv2.imshow("align" , image)
    facewidth = 512
    faceheight = 512
    eyeWidth = 0.70
    leftEyePts = shape[36:42]
    rightEyePts = shape[43:47]
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
     # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    # SHOULD BE 2.0
    desiredRightEyeX = 1.0 - eyeWidth
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - eyeWidth)
    desiredDist *= facewidth
    scale = desiredDist / dist
    print(scale) 
    # scale = -0.6
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # print(M)
    # update the translation component of the matrix
    tX = facewidth * 1.0
    tY = faceheight * 1.2
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    (w, h) = (facewidth, faceheight)
    output = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC)
    # return the aligned face
    return output

def run():
    # while(True):
    # ret, frame = cap.read()
    frame = cv2.imread("download.jpg")
    normal = cv2.imread("download.jpg")
    global detector
    dets = detector(frame, 1)
    shape = None
    # print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print(k)
        # rect = dlib.rectangle(d.left()-400 , d.top() - 200, d.right() + 400, d.bottom()+600)
        # d = rect
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        # k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)
        # print(shape)
        shape = shape_to_np(shape)
        teller = 0
        for (x, y) in shape:
            r = 0
            if (teller > 35 and teller < 43) or (teller > 42 and teller < 48):
                r=255
            cv2.circle(normal, (x, y), 1, (r, 0, 255), -1)
            teller = teller + 1
        x=d.left()
        y=d.top()
        w=d.right() -d.left()
        h=d.bottom()-d.top()
        crop_img = frame[y:y+h, x:x+w]
        frame = align(crop_img,shape)


    dets = detector(frame,1)
    for k,d in enumerate(dets):
        shape = predictor(frame, d)
        # print(shape)
        shape = shape_to_np(shape)
        print(shape[39:40][0][1])
        print(shape[42:43][0][1])
        # print(shape)
        # print(len(shape))
        teller = 0
        for (x, y) in shape:
            r = 0
            if (teller > 35 and teller < 43) or (teller > 42 and teller < 48):
                r=255
            cv2.circle(frame, (x, y), 1, (r, 0, 255), -1)
            teller = teller + 1

        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              # shape.part(1)))
    cv2.imshow("Normal", normal)
    cv2.imshow("Alligned", frame)
    key = cv2.waitKey(0) & 0xFF

# if the `q` key was pressed, break from the loop
    # if key == ord("q"):
    #     break

run()