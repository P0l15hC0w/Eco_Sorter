import cv2
import numpy as np
from teachable_machine import TeachableMachine
from pathlib import Path

camera = cv2.VideoCapture(1)

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = BASE_DIR / "models/hand.h5"
HAND_LABEL_PATH = BASE_DIR / "models/hand.txt"

hand_model = TeachableMachine(model_path=HAND_MODEL_PATH, labels_file_path=HAND_LABEL_PATH)
with open(HAND_LABEL_PATH, "r") as f: labels = [label.strip("\n") for label in f.readlines()]

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Otwarcie kamery nieudane.")
            break

        image = np.expand_dims((
                    np.asarray(
                    cv2.cvtColor(
                        cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB),
                    dtype=np.float32) / 127.5) - 1,
                axis=0)

        prediction_hand = hand_model._model.predict(image, verbose=0)

        print([[labels[i], np.round(pred, 3)] for i, pred in enumerate(prediction_hand[0])])

        cv2.imshow("Kamera", frame)
        
        if cv2.waitKey(1) == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()