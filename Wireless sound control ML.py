import math
import numpy as np
import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while cap.isOpened():
    state, img = cap.read()

    # processes the image for hand and stores the image
    result_img = hands.process(img)

    # Processing the image for the landmark points of multi hand
    if result_img.multi_hand_landmarks:
        for handLms in result_img.multi_hand_landmarks:
            landmarks_list = list()
            for id, landmark in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                landmarks_list.append([id, cx, cy])

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if landmarks_list:
                x_thumb, y_thumb = landmarks_list[4][1], landmarks_list[4][2]
                x_index, y_index = landmarks_list[8][1], landmarks_list[8][2]

                cv2.circle(img, (x_thumb, y_thumb), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x_index, y_index), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x_thumb, y_thumb), (x_index, y_index), (0, 0, 255), 3)

                z1 = (x_thumb + x_index) // 2
                z2 = (y_thumb + y_index) // 2

                line_length = math.hypot(x_index-x_thumb, y_index-y_thumb)

                if line_length < 50:
                    cv2.circle(img, (z1, z2), 10, (0, 255, 255), cv2.FILLED)

            volBar = np.interp(line_length, [50, 300], [400, 150])
            volPer = np.interp(line_length, [50, 300], [0, 100])

            vol = np.interp(line_length, [50, 300], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Volume container box
            cv2.rectangle(img, (50, 100), (85, 400), (128, 128, 128), 6)

            # Volume level
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (128, 128, 128), cv2.FILLED)

            # Numerical Volume
            cv2.putText(img, f'{int(volPer)}%', (100, 150), cv2.FONT_ITALIC, 2, (255, 255, 255), 2)

    # Navigation text
    cv2.putText(img, "Press Q to exit the program", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Video Capture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

