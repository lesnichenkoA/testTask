import numpy as np
import cv2


def median(frame):
    """Рассчитаем медиану в нашем окне, так как это самый оптимальный статистический параметр."""
    return np.median(frames, axis=0).astype(dtype=np.uint8)


cap = cv2.VideoCapture('test.mp4')

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

cv2.imshow('frame', median(frame))
cv2.waitKey(0)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

grayMedianFrame = cv2.cvtColor(median(frame), cv2.COLOR_BGR2GRAY)

ret = True
while (ret):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow('frame', dframe)
    cv2.waitKey(20)

cap.release()
cv2.destroyAllWindows()
