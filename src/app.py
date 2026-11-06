import cv2
import numpy as np
from teachable_machine import TeachableMachine
from pathlib import Path
import serial

COM_PORT = "COM3"

def read_cam(camera: cv2.VideoCapture):
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

def construct_packet(xy: tuple, indeks: int, stop: bool):
    return f"{xy[0]},{xy[1]},{indeks},{stop};"

try: 
    tries = 0
    while not com:
        com = serial.Serial(COM_PORT, 115200, timeout=1)
        tries += 1
        if tries > 10: raise
    
    running = True
    camera = cv2.VideoCapture(0)
    try:
        ret, _ = camera.read()
        if not ret:
            for index in range(1, 5):
                camera = cv2.VideoCapture(index)
                ret, _ = camera.read()
                if ret: camera.release(); raise

        BASE_DIR = Path(__file__).resolve().parent
        HAND_MODEL_PATH = BASE_DIR / "models/hand.h5"
        HAND_LABEL_PATH = BASE_DIR / "models/hand.txt"

        WASTE_MODEL_PATH = BASE_DIR / "models/waste.h5"
        WASTE_LABEL_PATH = BASE_DIR / "models/waste.txt"

        hand_model = TeachableMachine(model_path=HAND_MODEL_PATH, labels_file_path=HAND_LABEL_PATH)
        with open(HAND_LABEL_PATH, "r") as f: hand_labels = [label.strip("\n") for label in f.readlines()]

        waste_model = TeachableMachine(model_path=WASTE_MODEL_PATH, labels_file_path=WASTE_LABEL_PATH)
        with open(WASTE_LABEL_PATH, "r") as f: waste_labels = [label.strip("\n") for label in f.readlines()]

        try:
            while running:
                verdict = 0
                for i in range(5):
                    image, frame = read_cam(camera)
                    
                    prediction_hand = hand_model._model.predict(image, verbose=0)

                    verdict += float(prediction_hand[0])
                    print([[hand_labels[i], float(pred, 3)] for i, pred in enumerate(prediction_hand[0])])

                    cv2.imshow("Kamera", frame)
                    
                    if cv2.waitKey(1) == 27:
                        running = False
                        break

                vedrict /= 5

                if verdict < 15 and running:
                    image, frame = read_cam(camera)

                    prediction_trash_class = waste_model._model.predict()

                    trash_class = np.argmax(prediction_trash_class[0])

                # packet = construct_packet((1, 2), trash_class, stop)


        finally:
            camera.release()
            cv2.destroyAllWindows()
            exit(0)
    except:
        print("Nie znaleziono kamery")
        exit(1)
except:
    print("Port serial niedostepny")
    exit(1)

