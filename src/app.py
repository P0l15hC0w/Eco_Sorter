import cv2
import numpy as np
from teachable_machine import TeachableMachine
from pathlib import Path

def read_cam(camera):
    ret, frame = camera.read()
    
    if not ret:
        print("Otwarcie kamery nieudane.")
        raise
    
    image = np.expand_dims((
            np.asarray(
                cv2.cvtColor(
                    cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB),
                dtype=np.float32) / 127.5) - 1,
            axis=0)

    return image, frame
    

camera = cv2.VideoCapture(0)
try:
    ret, _ = camera.read()
    if not ret:
        for index in range(1, 5):
            camera = cv2.VideoCapture(index)
            ret, _ = camera.read()
            if ret: raise

    BASE_DIR = Path(__file__).resolve().parent
    HAND_MODEL_PATH = BASE_DIR / "models/hand.h5"
    HAND_LABEL_PATH = BASE_DIR / "models/hand.txt"

    WASTE_MODEL_PATH = BASE_DIR / "models/waste.h5"
    WASTE_LABEL_PATH = BASE_DIR / "models/waste.txt"

    hand_model = TeachableMachine(model_path=HAND_MODEL_PATH, labels_file_path=HAND_LABEL_PATH)
    with open(HAND_LABEL_PATH, "r") as f: labels = [label.strip("\n") for label in f.readlines()]

    hand_model = TeachableMachine(model_path=HAND_MODEL_PATH, labels_file_path=HAND_LABEL_PATH)
    with open(HAND_LABEL_PATH, "r") as f: labels = [label.strip("\n") for label in f.readlines()]

    try:
        while True:
            verdict = 0
            for i in range(5):
                image, frame = read_cam(camera)
                
                prediction_hand = hand_model._model.predict(image, verbose=0)

                verdict += float(prediction_hand[0])
                print([[labels[i], float(pred, 3)] for i, pred in enumerate(prediction_hand[0])])

                cv2.imshow("Kamera", frame)
                
                if cv2.waitKey(1) == 27:
                    break

            vedrict /= 5

            if verdict < 15:
                pass
                """
                trash_class = 0
                for i in range(5):
                    image, frame = read_cam(camera)

                    prediction_trash_class = waste_model._model.predict()
                """
    finally:
        camera.release()
        cv2.destroyAllWindows()
except:
    print("Nie znaleziono kamery")
