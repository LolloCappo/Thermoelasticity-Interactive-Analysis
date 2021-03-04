import numpy as np
import matplotlib.pyplot as plt
from flirpy.camera.lepton import Lepton
import time
import cv2

scale_factor = 3

camera = Lepton()
image = camera.grab()
camera.close()

#image = cv2.imdecode(camera.grab(), cv2.IMREAD_COLOR)
print(image.shape)
(y_or,x_or) = image.shape
image_roi = np.zeros((x_or, y_or))
image_roi = cv2.normalize(image,image_roi, 0, 255, cv2.NORM_MINMAX)
image_roi = cv2.flip(image_roi, 0)
image_roi = cv2.resize(image_roi, (x_or*scale_factor,y_or*scale_factor), interpolation = cv2.INTER_AREA)
print(image_roi)
size_roi = cv2.selectROI(image_roi.astype('uint8'))
print(camera.frame_count)
print(camera.frame_mean)
print(camera.ffc_temp_k)
print(camera.fpa_temp_k)


cv2.destroyAllWindows()
data = []
frame = np.zeros((x_or, y_or))#np.zeros((image.shape[1], image.shape[0]))

while True:

    camera = Lepton()

    t1 = time.time()

    image = camera.grab()
    data.append(image[int(size_roi[1]):int(size_roi[1]+size_roi[3]), int(size_roi[0]):int(size_roi[0]+size_roi[2])])
    frame = cv2.normalize(image,  frame, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.flip(frame, 0)

    frame = cv2.resize(frame, (x_or*scale_factor,y_or*scale_factor), interpolation = cv2.INTER_AREA)

    #bgr_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    #im_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    cv2.imshow('live',frame.astype('uint8'))

    k = cv2.waitKey(33)
    if k==27: 
        break
    
    camera.close()

    #print(1/(time.time()-t1))

    time.sleep(0.1)

cv2.destroyAllWindows()
camera.close()

data = np.stack(data)
media = []

for frame in data:
    media.append(np.mean(frame))
plt.plot(np.array(media))
plt.show()



