import cv2
import time
import mediapipe as mp

class FaceDetector():
    def __init__(self, min_conf=0.5, model=0):
        self.min_conf = min_conf
        self.model = model

        #initializations of mediapipe libs
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_conf, self.model)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw = True):
        scaleFactor = 30
        n_w = int(img.shape[1]*scaleFactor/100)
        n_h = int(img.shape[0]*scaleFactor/100)
        dim = (n_w,n_h)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.fd_results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.fd_results.detections:
            for face_no, face in enumerate(self.fd_results.detections):
                bboxC = face.location_data.relative_bounding_box
                ih, iw, c = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, face.score])
                img = self.fancyDraw(img,bbox)
                #text
                cv2.putText(img, f'{int(face.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0,255,0), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=10):
        x,y,w,h = bbox
        x1,y1 = x+w,y+h
        cv2.line(img, (x,y), (x+l,y), (255,0,255),t)
        cv2.line(img, (x,y), (x,y+l), (255,0,255),t)

        cv2.line(img, (x+w, y), (x+w-l, y), (255, 0, 255), t)
        cv2.line(img, (x+w, y), (x+w, y+l), (255, 0, 255), t)
        #
        cv2.line(img, (x, y+h), (x, y+h-l), (255, 0, 255), t)
        cv2.line(img, (x, y+h), (x+l, y+h), (255, 0, 255), t)
        #
        cv2.line(img, (x+w, y+h), (x+w-l, y+h), (255, 0, 255), t)
        cv2.line(img, (x+w, y+h), (x+w, y+h-l), (255, 0, 255), t)

        # now lets draw the rectangle and put the detection score on the img
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        return img

def main():
    cap = cv2.VideoCapture('videos/discuss.mp4')
    pTime = 0
    fd = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = fd.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime -pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {str(int(fps))}', (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


