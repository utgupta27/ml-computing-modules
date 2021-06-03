import cv2
import mediapipe
import time


class HandDetection:
    def __init__(self, markLandmarks=list(range(21)), drawPalm=True, mark=True):
        self.mark = mark
        self.delayFrame = 1
        self.prevTime = 0
        self.currTime = 0
        self.drawPalm = drawPalm
        self.stream = cv2.VideoCapture(0)
        self.markLandmarks = markLandmarks
        self.mediapipeHands = mediapipe.solutions.hands
        self.hands = self.mediapipeHands.Hands(max_num_hands=1)
        self.mediapipeDraw = mediapipe.solutions.drawing_utils

    def markPoints(self, frame):
        markList = []
        for index, landmrk in enumerate(self.handLandmarks.landmark):
            if index in self.markLandmarks:
                height, width, c = frame.shape
                pixelX, pixelY = int(landmrk.x * width), int(landmrk.y * height)
                if self.mark:
                    cv2.circle(frame, (pixelX, pixelY), 10, (100, 255, 0), cv2.FILLED)
                markList.append([index, pixelX, pixelY])
        return markList

    def detectHand(self, frame):
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processedImageResults = self.hands.process(rgbImg)
        if processedImageResults.multi_hand_landmarks:
            for self.handLandmarks in processedImageResults.multi_hand_landmarks:
                # print(processedImageResults.multi_hand_landmarks)
                print(self.markPoints(frame))
                if self.drawPalm:
                    self.mediapipeDraw.draw_landmarks(frame, self.handLandmarks,
                                                      self.mediapipeHands.HAND_CONNECTIONS)
        return frame

    def main(self):
        while True:
            success, self.frame = self.stream.read()
            self.frame = cv2.flip(self.frame, flipCode=1)

            '''
                Perform your computation on each frame of live video Below
            '''
            self.frame = self.detectHand(self.frame)

            # End

            self.currTime = time.time()
            framesPerSecond = 1 / (self.currTime - self.prevTime)
            self.prevTime = self.currTime

            cv2.putText(self.frame, "fps" + str(int(framesPerSecond)), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Palm Detection", self.frame)
            if cv2.waitKey(self.delayFrame) & 0xff == ord(' '):
                break


if __name__ == '__main__':
    '''
        See the image PATH: handDetection/res/hand_landmarks.png to get an idea to 
        display that particular section of hand marking and print its locations.
        list:"markLandmarks" contains any value in range(21).
        Ex: markLandmarks = [4,8,12,16,20] 
    '''
    webCamStream = HandDetection(markLandmarks=[4, 8, 12, 16, 20], drawPalm=True, mark=False)
    webCamStream.main()
