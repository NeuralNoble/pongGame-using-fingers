import cv2
import streamlit as st
import numpy as np
import mediapipe as mp

# Set up camera
width = 1280
height = 720
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

class mpHands:
    def __init__(self, maxHands=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=minDetectionConfidence,
            min_tracking_confidence=minTrackingConfidence
        )

    def marks(self, frame):
        myHands = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                myHand = []
                for landmark in handLms.landmark:
                    myHand.append((int(landmark.x * width), int(landmark.y * height)))
                myHands.append(myHand)
        return myHands

boxColor = (0, 0, 255)
paddleWidth = 125
paddleHeight = 25
ballRadius = 15
ballColor = (255, 0, 0)
xPos = width // 2
yPos = height // 2
deltax = 4
deltay = 4
lives = 5
score = 0
font = cv2.FONT_HERSHEY_SIMPLEX

findHands = mpHands()

st.title('Pong Game with Hand Gestures')
start_button = st.button('Start')
stop_button = st.button('Stop')
score_display = st.empty()
lives_display = st.empty()
frame_display = st.empty()

run = False

if start_button:
    run = True
if stop_button:
    run = False

while True:
    if run:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        cv2.circle(frame, (xPos, yPos), ballRadius, ballColor, -1)
        cv2.putText(frame, str(score), (25, int(6 * paddleHeight)), font, 5, (0, 0, 255), 5)
        cv2.putText(frame, str(lives), (width - 125, int(6 * paddleHeight)), font, 5, (0, 0, 255), 5)
        myHands = findHands.marks(frame)

        if myHands:
            hand = myHands[0]
            cv2.rectangle(frame, (hand[8][0] - paddleWidth // 2, 0), (hand[8][0] + paddleWidth // 2, paddleHeight), boxColor, -1)

            topEdge = yPos - ballRadius
            bottomEdge = yPos + ballRadius
            leftEdge = xPos - ballRadius
            rightEdge = xPos + ballRadius

            if leftEdge <= 0 or rightEdge >= width:
                deltax *= -1
            if bottomEdge >= height:
                deltay *= -1

            if topEdge <= paddleHeight:
                if hand[8][0] - paddleWidth // 2 <= xPos <= hand[8][0] + paddleWidth // 2:
                    deltay *= -1
                    score += 1
                    if score in [5, 10, 15, 20]:
                        deltax *= 2
                        deltay *= 2
                else:
                    xPos = width // 2
                    yPos = height // 2
                    lives -= 1

            xPos += deltax
            yPos += deltay

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame)
        score_display.text(f"Score: {score}")
        lives_display.text(f"Lives: {lives}")

        if lives == 0:
            st.write("Game Over!")
            break

        cv2.waitKey(1)
    else:
        break

cam.release()
cv2.destroyAllWindows()
