
import cv2
import numpy as np
import os
import dlib
import math
import sys
import time
import matplotlib.pyplot as plt

from skimage import data, transform

# from imutils import face_utils
subjects  =["","Fabian Mijsters","Random Person"]

global face_cascade, predictor,clahe,detector, angle
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
angle = 0


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

def detectPose(im,image_points):
    size = im.shape
    global angle
#2D image points. If you change the image, you need to change vector
    # image_points = np.array([
    #                             (359, 391),     # Nose tip
    #                             (399, 561),     # Chin
    #                             (337, 297),     # Left eye left corner
    #                             (513, 301),     # Right eye right corne
    #                             (345, 465),     # Left Mouth corner
    #                             (453, 469)      # Right mouth corner
    #                         ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    # print("Camera Matrix :\n {0}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    

    # print("Rotation Vector:\n {0}".format(rotation_vector))
    # print( "Translation Vector:\n {0}".format(translation_vector))
    # print(math.tan(rotation_vector[0]/rotation_vector[1]))
    angle = angle + math.tan(rotation_vector[0]/rotation_vector[1])
    # print(angle)
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
     
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    if test:
        delta_x = p1[0] - p2[0]
        delta_y = p1[1] - p2[1]
        theta_radians = math.atan2(delta_y, delta_x) 
        theta_radians = math.tan(rotation_vector[1]/rotation_vector[0])
        theta = theta_radians
        tx = 0
        ty = 0

        S, C = np.sin(theta), np.cos(theta)

        H = np.array([[C, -S, tx],
              [S,  C, ty],
              [0,  0, 1]])
        
        r, c = im.shape[0:2]

        T = np.array([[1, 0, -c / 2.],
                      [0, 1, -r / 2.],
                      [0, 0, 1]])
        # print(translation_vector)
        S = np.array([[1, 0, 0],
                      [0, 1.3, 0],
                      [0, 1e-3, 1]])

        # img_rot = transform.ProjectiveTransform(im, H)
        trans = transform.ProjectiveTransform(H)
        img_rot_center_skew = transform.warp(im, transform.ProjectiveTransform(S.dot(np.linalg.inv(T).dot(H).dot(T))))
        img_rot = transform.warp(im, trans)
        f, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        ax1.imshow(img_rot, cmap=plt.cm.gray, interpolation='nearest')
        ax2.imshow(img_rot_center_skew, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()
        print(math.degrees(theta_radians))



    # print(math.tan(p2[0]/p1[0]))
    # print(math.tan(p2[1]/p1[1]))
    # print(str(p1) + " - " + str(p2))
    # print(p2) 
    # print(p2)
    cv2.line(im, p1, p2, (255,0,0), 2)
    return im

def align(image,shape): 
    # cv2.imshow("align" , image)
    sizeIncrement = 1
    facewidth = 256 * sizeIncrement
    faceheight = 256 * sizeIncrement
    eyeWidth = 0.35 
    leftEyePts = shape[36:42]
    rightEyePts = shape[42:48]
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    #might need -180
    angle = np.degrees(np.arctan2(dY, dX)) 
     # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    # SHOULD BE 2.0
    desiredRightEyeX = (1.0) - eyeWidth
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = (np.sqrt((dX ** 2) + (dY ** 2)) * sizeIncrement)

    desiredDist = (desiredRightEyeX - eyeWidth)

    desiredDist *= facewidth

    scale = desiredDist / dist
    # print(scale) 
    # scale = -0.6
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # print(M)
    # update the translation component of the matrix
    tX = facewidth * (0.5)
    tY = faceheight * eyeWidth 
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    (w, h) = (facewidth, faceheight)
    output = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC)
    # return the aligned face
    return output

def fix_the_alignment(frame,normal):
    global detector
    dets = detector(frame, 1)
    shape = None

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # print(k)
        # rect = dlib.rectangle(d.left()-400 , d.top() - 200, d.right() + 400, d.bottom()+600)
        # d = rect
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        # k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)
        # print(shape)
        shape = shape_to_np(shape)
        print("ratio between left eye and middle of head is", str(((shape[39][0]- shape[36][0]) * 100) / (shape[16][0] - shape[0][0])))
        plot1 = shape[39]
        plot2 = shape[36]
        plot3 = shape[48]
        plot4 = shape[55]
        euclidean_distance = math.sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
        euclidean_distance2 = math.sqrt( (plot3[0]-plot4[0])**2 + (plot3[1]-plot4[1])**2 )

        print("ratio between left eye and middle of head is euclidean_distance", str(((euclidean_distance) * 100) / (euclidean_distance2)))
        print(euclidean_distance)
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
        # crop_img = frame[y:y+h, x:x+w]
        frame = align(frame,shape)
        return frame , normal

def run(allign, pose):
    while(True):
        ret, frame = cap.read()
        normal = frame
    # frame = cv2.imread("download.jpg")
    # normal = cv2.imread("download.jpg")
        

        global detector
        # uncomment to keep face straight when it tilts WIP
        if allign:
            frame,normal = fix_the_alignment(frame,normal)


        dets = detector(frame,1)
        for k,d in enumerate(dets):
            shape = predictor(frame, d)
            # print(shape)
            shape = shape_to_np(shape)
            # print(shape[39:40][0][1])
            # print(shape[42:43][0][1])
            # print(shape)
            # return
            # print(shape)
            # print(len(shape))
            # print("Ratio between left eye and middle of head is", str(((shape[39][0]- shape[36][0]) * 100) / (shape[16][0] - shape[0][0])))
            teller = 0
            for (x, y) in shape:
                r = 0
                if (teller > 35 and teller < 43) or (teller > 42 and teller < 48):
                    r=255
                cv2.circle(frame, (x, y), 1, (r, 0, 255), -1)


                # cv2.line(frame,(shape[0][0],shape[0][1]),(shape[16][0],shape[16][1]),(r,0,255),1)
                # cv2.line(frame,(shape[36][0],shape[36][1]),(shape[39][0],shape[39][1]),(255,0,0),1)

                # print(shape[16][0] - shape[0][0])
                # print(shape[39][0] - shape[36][0])
                

                teller = teller + 1

            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  # shape.part(1)))
        # cv2.imshow("Normal", normal)


        #Uncomment 2 lines below to enable pose detection
        if pose:
            image_points = np.array([
                                        shape[30],     # Nose tip
                                        shape[8],     # Chin
                                        shape[36],     # Left eye left corner
                                        shape[45],     # Right eye right corne
                                        shape[48],     # Left Mouth corner
                                        shape[54]      # Right mouth corner
                                    ], dtype="double")

            frame = detectPose(frame,image_points)

        cv2.imshow("Detection", frame)
        key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

allign = False
pose = False
print(sys.argv)
if "allign" in sys.argv:
    allign = True
if "pose" in sys.argv:
    pose = True
run(allign,pose)