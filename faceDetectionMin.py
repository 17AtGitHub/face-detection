import cv2
import mediapipe as mp
import time

# m is using external videos instead of webcam because
# in webcam there will be a limitation of frame rate

cap = cv2.VideoCapture('videos/discuss.mp4')

# getting the modules from the imported libraries

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()

    # image resizing
    scaleFactor = 30
    width = int(img.shape[1]*scaleFactor/100)
    height = int(img.shape[0]*scaleFactor/100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # to process this image, need to convert it to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fd_results = faceDetection.process(imgRGB)
    # the detections parameter in the fd_results object is used to check if any face detected or not
    if fd_results.detections:
        for face_no, face in enumerate(fd_results.detections):
            bboxC = face.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox, (255,0,255), 3)
            # there are all the kinds of different 7 i think deetected on the face
            # we are now going to locate the bounding box by its coordinatees

            # next, I would also like to get the detection confidence score beside the rectangle
            cv2.putText(img, f'{int(face.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,255,0), 2)
            # print(f'FACE NUMBER: {face_no+1}')
            # print('===================')
            # print('Detection Confidence: {round(face.score[0],2)}')
            # face_data = face.location_data
            # print(f'Face Bounding Box:n{face_data.relative_bounding_box}')
            # for i in range(2):
            #     print(f'{mpFaceDetection.FaceKeyPoint(i).name}:')
            #     print(f'{face_data.relative_keypoints[mpFaceDetection.FaceKeyPoint(i).value]}')
            


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (int(width/7),int(height/7)), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

