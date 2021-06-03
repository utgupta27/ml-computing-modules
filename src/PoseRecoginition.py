import cv2 as cv
import mediapipe
import time


class PoseDetection():
    def __init__(self, markLandmarks=list(range(21)), drawPose=True, mark=True) -> None:
        self.stream = cv.VideoCapture(0)
        self.drawPose = drawPose
        self.mark = mark
        self.delayFrame = 1
        self.prevTime = 0
        self.currTime = 0
        self.markLandmarks = markLandmarks
        self.mediapipePose = mediapipe.solutions.pose
        self.pose = self.mediapipePose.Pose()
        self.mediapipeDraw = mediapipe.solutions.drawing_utils

    def markPoint(self):
        markList = []
        for index, landmrk in enumerate(self.processedImageResults.pose_landmarks.landmark):
            if index in self.markLandmarks:
                height, width, c = self.frame.shape
                pixelX, pixelY = int(landmrk.x * width), int(landmrk.y * height)
                if self.mark:
                    cv.circle(self.frame, (pixelX, pixelY), 10, (100, 255, 0), cv.FILLED)
                markList.append([index, pixelX, pixelY])
        return markList

    def detectPose(self):
        rgbFrame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        self.processedImageResults = self.pose.process(rgbFrame)
        if self.processedImageResults.pose_landmarks:
            print(self.markPoint())
            if self.drawPose:
                self.mediapipeDraw.draw_landmarks(self.frame, self.processedImageResults.pose_landmarks,
                                                  self.mediapipePose.POSE_CONNECTIONS)

    def main(self):
        while True:
            success, self.frame = self.stream.read()
            self.frame = cv.flip(self.frame, flipCode=1)

            '''
                Perform your computation on each frame of live video Below
            '''
            self.detectPose()

            # End

            self.currTime = time.time()
            framesPerSecond = 1 / (self.currTime - self.prevTime)
            self.prevTime = self.currTime

            cv.putText(self.frame, "fps" + str(int(framesPerSecond)), (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.imshow("Pose Detection", self.frame)
            if cv.waitKey(self.delayFrame) & 0xff == ord(' '):
                break


if __name__ == "__main__":
    obj = PoseDetection(markLandmarks=[0, 11, 12, 18, 20, 17, 19], drawPose=True, mark=True)
    obj.main()
