import cv2
import numpy as np
from teachable_machine import TeachableMachine
from pathlib import Path

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

    image = np.expand_dims(display_frame - 1, axis=0)
    
    return image, display_frame

camera = cv2.VideoCapture(1)
running = True

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models/hand.h5"
HAND_LABEL_PATH = BASE_DIR / "models/hand.txt"

WASTE_MODEL_PATH = BASE_DIR / "models/waste.h5"
WASTE_LABEL_PATH = BASE_DIR / "models/waste.txt"

LOCATION_Y_MODEL_PATH = BASE_DIR / "models/location_y.h5"
LOCATION_Y_LABEL_PATH = BASE_DIR / "models/location_y.txt"

LOCATION_X_MODEL_PATH = BASE_DIR / "models/location_x.h5"
LOCATION_X_LABEL_PATH = BASE_DIR / "models/location_x.txt"

try:
    try: 
        hand_model = TeachableMachine(model_path=HAND_MODEL_PATH, labels_file_path=HAND_LABEL_PATH)
        with open(HAND_LABEL_PATH, "r") as f: hand_labels = [label.strip("\n") for label in f.readlines()]
    except: raise Exception("Otwarcie modelu dloni niemozliwe")
        
    try:
        waste_model = TeachableMachine(model_path=WASTE_MODEL_PATH, labels_file_path=WASTE_LABEL_PATH)
        with open(WASTE_LABEL_PATH, "r") as f: waste_labels = [label.strip("\n") for label in f.readlines()]
    except: raise Exception("Otwarcie modelu odpadow niemozliwe")
        
    try:
        location_x_model = TeachableMachine(model_path=LOCATION_X_MODEL_PATH, labels_file_path=LOCATION_X_LABEL_PATH)
        with open(LOCATION_X_LABEL_PATH, "r") as f: location_x_labels = [label.strip("\n") for label in f.readlines()]
    except: raise Exception("Otwarcie modelu lokalizacji osi x odpadu niemozliwe")
        
    try:
        location_y_model = TeachableMachine(model_path=LOCATION_Y_MODEL_PATH, labels_file_path=LOCATION_Y_LABEL_PATH)
        with open(LOCATION_Y_LABEL_PATH, "r") as f: location_y_labels = [label.strip("\n") for label in f.readlines()]
    except: raise Exception("Otwarcie modelu lokalizacji osi y odpadu niemozliwe")

    while running:
        image, display_frame = read_cam(camera)

        prediction_hand = hand_model._model.predict(image, verbose=0)[0]

        if prediction_hand[0] < 0.15 and prediction_hand[2] > 0.4:
            prediction_waste = waste_model._model.predict(image, verbose=0)[0]
            waste_class_idx = np.argmax(prediction_waste)

            if waste_class_idx != 0:
                prediction_location_x = np.array(location_x_model._model.predict(image, verbose=0)[0]).tolist()
                prediction_location_y = np.array(location_y_model._model.predict(image, verbose=0)[0]).tolist()
                
                x = [prediction_location_x.index(el) for el in prediction_location_x if el > 0.2]
                y = [prediction_location_y.index(el) for el in prediction_location_y if el > 0.2]

                print(waste_labels[waste_class_idx], f"x{x}", f"y{y}")
        else:
            if prediction_hand[0] >= 0.15:
                print(hand_labels[0])
            else:
                print(hand_labels[1])

        cv2.imshow("Kamera", display_frame)
                            
        if cv2.waitKey(1) == 27:
            running = False
            break

except Exception as e:
    print(f"Exception: {e}")

finally:
    camera.release()
    cv2.destroyAllWindows()