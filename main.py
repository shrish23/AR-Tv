import cv2
import numpy as np
import cv2.aruco as aruco

def findArucoMarkers(img, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParams)

    if draw:
        aruco.drawDetectedMarkers(img, corners, ids)
    
    return [corners, ids]

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def augmentAruco(bbox, img, imgAug):
    if len(bbox) != 4:
        return img

    # Assuming markers are roughly placed in top-left, top-right, bottom-right, bottom-left order
    pts1 = np.array([c[0] for c in bbox], dtype="float32")
    pts1 = orderPoints(pts1)

    h, w, c = imgAug.shape
    pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts1.astype(int), (255,) * img.shape[2])
    maskInv = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(img, maskInv)
    imgOut = cv2.bitwise_or(img, imgOut)

    return imgOut

def drawRectangleFromAruco(bbox, img):
    if len(bbox) != 4:
        return img

    # Assuming markers are roughly placed in top-left, top-right, bottom-right, bottom-left order
    pts1 = np.array([c[0] for c in bbox], dtype="float32")
    pts1 = orderPoints(pts1)

    # Draw the polygon connecting the ArUco markers
    cv2.polylines(img, [pts1.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=5)

    return img

# Initialize webcam and read the overlay image
cap = cv2.VideoCapture(1)
videoAug = cv2.VideoCapture("tennis.mp4")

while True:
    ret, img = cap.read()
    retVid, vid = videoAug.read()
    if not ret:
        break

    if not retVid:
        videoAug.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    arucoFound = findArucoMarkers(img)

    # Check if exactly four markers are found
    if arucoFound[1] is not None and len(arucoFound[0]) == 4:
        bboxs = [bbox[0] for bbox in arucoFound[0]]
        img = augmentAruco(bboxs, img, vid)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
