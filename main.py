'''                                                                        ***********   ************  ***********     **********  
This Code is designed by:                                                  ************  ************   ************   **********
    Name: Abdelrahman Wael Ammar                                           *****     *** ***      ***   *****    ****     ****
    Phone: +201010331884                                                   *****     *** ***      ***   *****    ****     ****
    E-Mail: bodiwael999@gmail.com                                          ************* ***      ***   *****    ****     ****
    Linkedin: https://www.linkedin.com/in/abdelrahman-wael-ammar/          ******    *** ***      ***   *****    ****     ****
    Github: https://github.com/bodiwael                                    ************  ************   ************   **********
    Fiverr: https://www.fiverr.com/abderlahmanwael                         ***********   ************  ***********     **********
'''

# Coding Library Importing

import cv2 as cv            # OpenCV Library for Image Visuallization
import numpy as np          # Numpy Library
import mediapipe as mp      # Mediapipe Library for Face Detection Mesh
import csv                  # CSV Library for File Reading
import copy                 # Copy Library for path localization
import itertools            # IterTools Library

from model import KeyPointClassifier        # Face emotion machine Learning model

#**************************************************************************

# Face Mesh Defining Style

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(max_num_faces=10,static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:

        # Outer rectangle

        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255, 0, 0), 7)

    return image

def calc_bounding_rect(image, landmarks):

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):

        landmark_x = min(int(landmark.x * image_width), image_width - 1)

        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 255, 0), 5)

    if facial_text != "":

        info_text = 'Emotion : ' + facial_text

        print(info_text)

    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)

    return image

cap_device = 0
# cap_width = 1920
# cap_height = 1080
cap_width = 1080
cap_height = 640


use_brect = True

# Camera preparation

cap = cv.VideoCapture(cap_device, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

# Model load

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()


# Read labels

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0

while True:

    # Process Key (ESC: end)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    # Camera capture
    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            #emotion classification
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            
            draw.draw_landmarks(debug_image, face_landmarks, facmesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))
            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(
                    debug_image,
                    brect,
                    keypoint_classifier_labels[facial_emotion_id])

    # Screen reflection
    cv.imshow('Facial Emotion Recognition', debug_image)
#     cv.imshow('Original', image)

cap.release()
cv.destroyAllWindows()