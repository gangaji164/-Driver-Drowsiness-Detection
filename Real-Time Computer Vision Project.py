import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
thresh = 0.25
frame_check = 20


(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
detector = dlib.get_frontal_face_detector()     
predictor = dlib.shape_predictor(r"C:\Users\ganga\OneDrive\Desktop\charity\emotion\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)

flag=0



while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)
    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)


        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]


        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)


        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag+=1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame,"*******ALERT*******",(10,325),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                mixer.music.play()
        else:
            flag=0
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
