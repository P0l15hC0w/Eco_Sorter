import cv2
import numpy as np

def read_cam(camera: cv2.VideoCapture):
    ret, frame = camera.read()
    
    if not ret:
        print("Otwarcie kamery nieudane.")
        raise RuntimeError("Nie udalo sie odczytac obrazu z kamery.")
    
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped_frame = frame[top:top + crop_size, left:left + crop_size]
    display_frame = cv2.resize(
        np.asarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB), dtype=np.float32) / 127.5,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )
    
    return display_frame

camera = cv2.VideoCapture(1)
running = True

try:
    while running:
        display_frame = read_cam(camera)

        cv2.imshow("Kamera", display_frame)
                            
        if cv2.waitKey(1) == 27:
            running = False
            break
finally:
    camera.release()
    cv2.destroyAllWindows()

    