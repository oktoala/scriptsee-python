import pprint

import cv2
import os
import tabulate
import json
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    num_hands=2,
    base_options=BaseOptions(model_asset_path=model_path),
)

detector = HandLandmarker.create_from_options(options)


r = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',]
alphs = os.listdir('./Dataset/datatrain')

data_list = []

for alph in alphs:

    images = os.listdir(os.path.join('./Dataset/datatrain', alph))

    for img in images:
        print(alph + ' - ' + img)
        data = {}
        data['LABEL'] = alph;
        data['IMAGE_NAME'] = img
        
        mp_image = mp.Image.create_from_file(os.path.join('./Dataset/datatrain', alph, img))

        hand_landmarker_result = detector.detect(mp_image)

        handedness = hand_landmarker_result.handedness
        hand_landmarks = hand_landmarker_result.hand_landmarks

        for idx_hn in range(len(handedness)):
            category_name = handedness[idx_hn][0].category_name

            if (category_name == 'Right'):
                data['RIGHT_PROB'] = handedness[idx_hn][0].score
                for idx_hl in range(len(hand_landmarks[idx_hn])):
                    hand_landmark = hand_landmarks[idx_hn][idx_hl];
                    data[category_name + '_X_' +r[idx_hl] ] = hand_landmark.x
                    data[category_name + '_Y_' +r[idx_hl] ] = hand_landmark.y
                    data[category_name + '_Z_' +r[idx_hl] ] = hand_landmark.z
            elif (category_name == 'Left'):
                data['LEFT_PROB'] = handedness[idx_hn][0].score
                for idx_hl in range(len(hand_landmarks[idx_hn])):
                    hand_landmark = hand_landmarks[idx_hn][idx_hl];
                    data[category_name + '_X_' +r[idx_hl] ] = hand_landmark.x
                    data[category_name + '_Y_' +r[idx_hl] ] = hand_landmark.y
                    data[category_name + '_Z_' +r[idx_hl] ] = hand_landmark.z

        data_list.append(data)     




with open('./data.json', 'w') as f:
    json.dump(data_list, f, indent=4)

pprint.pprint(data_list)

# annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
# cv2.imshow("img", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

