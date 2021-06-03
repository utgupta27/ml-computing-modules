import cv2 as cv
import mediapipe
import time


class FaceDetection:
    def __init__(self) -> None:
        self.delayFrame = 1
        self.prevTime = 0
        self.currTime = 0
        self.stream = cv.VideoCapture(0)
        self.mediapipeFace = mediapipe.solutions.face_detection
        self.face = self.mediapipeFace.FaceDetection(0.50)
        self.mediapipeDraw = mediapipe.solutions.drawing_utils

    def detectFace(self):
        self.rgbFrame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        self.processedFrameResult = self.face.process(self.rgbFrame)
        bounds = []
        if self.processedFrameResult:
            if self.processedFrameResult.detections:
                for index, landmarkInfo in enumerate(self.processedFrameResult.detections):
                    # self.mediapipeDraw.draw_detection(self.frame,landmarkInfo)
                    # print(index,landmarkInfo)
                    boundingBoxRatio = landmarkInfo.location_data.relative_bounding_box
                    height, width, channel = self.frame.shape
                    boundingBoxDimension = int(boundingBoxRatio.xmin * width), \
                                           int(boundingBoxRatio.ymin * height), \
                                           int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width), \
                                           int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)
                    # print(landmarkInfo.score[index])
                    # print(boundingBoxDimension)
                    cv.putText(self.frame, str(int(landmarkInfo.score[0] * 100)) + "%", \
                               (int(boundingBoxRatio.xmin * width), int(boundingBoxRatio.ymin * height) - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 225), 2)
                    # cv.rectangle(self.frame,boundingBoxDimension[0:2],boundingBoxDimension[2:4],(0,225,0),1)
                    cv.line(self.frame, (int(boundingBoxRatio.xmin * width), int(boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.xmin * width), int(boundingBoxRatio.ymin * height) + 25), (0, 255, 0),
                            3)
                    cv.line(self.frame, (int(boundingBoxRatio.xmin * width), int(boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.xmin * width) + 25, int(boundingBoxRatio.ymin * height)), (0, 255, 0),
                            3)

                    cv.line(self.frame, (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                             int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height) - 25), (0, 255, 0),
                            3)
                    cv.line(self.frame, (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width) - 25,
                             int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), (0, 255, 0), 3)

                    cv.line(self.frame, (int(boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.xmin * width),
                             int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height) - 25), (0, 255, 0),
                            3)
                    cv.line(self.frame, (int(boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.xmin * width) + 25,
                             int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height)), (0, 255, 0), 3)

                    cv.line(self.frame, (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                             int(boundingBoxRatio.ymin * height) + 25), (0, 255, 0), 3)
                    cv.line(self.frame, (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width),
                                         int(boundingBoxRatio.ymin * height)), \
                            (int(boundingBoxRatio.width * width + boundingBoxRatio.xmin * width) - 25,
                             int(boundingBoxRatio.ymin * height)), (0, 255, 0), 3)

                    cv.putText(self.frame, "Person " + str(index + 1), \
                               (int(boundingBoxRatio.xmin * width),
                                int(boundingBoxRatio.height * height + boundingBoxRatio.ymin * height) + 25),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
                    bounds.append([index, boundingBoxDimension])
                return bounds

    def main(self):
        while True:
            success, self.frame = self.stream.read()
            self.frame = cv.flip(self.frame, flipCode=1)

            '''
                Perform your computation on each frame of live video Below
            '''
            print(self.detectFace())

            # End

            self.currTime = time.time()
            framesPerSecond = 1 / (self.currTime - self.prevTime)
            self.prevTime = self.currTime

            cv.putText(self.frame, "fps " + str(int(framesPerSecond)), (5, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.imshow("Face Detection", self.frame)
            if cv.waitKey(self.delayFrame) & 0xff == ord(' '):
                break


if __name__ == '__main__':
    webCamStream = FaceDetection()
    webCamStream.main()
