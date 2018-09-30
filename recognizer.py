
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

def run():
    while(True):
        ret, frame = cap.read()
        global detector
        dets = detector(frame, 1)
        shape = None
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
                    # Get the landmarks/parts for the face in box d.
            shape = predictor(frame, d)
            shape = shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

run()