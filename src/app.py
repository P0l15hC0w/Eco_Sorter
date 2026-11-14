import cv2
import numpy as np
from teachable_machine import TeachableMachine
from pathlib import Path
from time import sleep
import serial

COM_PORT = "COM3"

def read_cam(camera: cv2.VideoCapture):
    ret, frame = camera.read()
    
    if not ret:
        print("Otwarcie kamery nieudane.")
        raise RuntimeError("Nie udalo sie odczytac obrazu z kamery.")
    
    image = np.expand_dims((
            np.asarray(
                cv2.cvtColor(
                    cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB),
                dtype=np.float32) / 127.5) - 1,
            axis=0)

    return image, frame

def construct_packet(xy: tuple, indeks, stop: bool):
    return f"{xy[0]},{xy[1]},{indeks},{stop};"

def send_packet(packet: str, com: serial.Serial):
    com.write(packet.encode("UTF-8"))

try: 
    tries = 1
    try: com = serial.Serial(COM_PORT, 115200, timeout=1)
    except: com = None

    while not com:
        sleep(0.25); print(f"Otwarcie portu serial {COM_PORT} nieudane, proba: {tries}")

        try: 
            com = serial.Serial(COM_PORT, 115200, timeout=1)
        except:
            tries += 1
            if tries > 10: raise Exception("Port serial niedostepny")

    running = True
    try:
        camera = cv2.VideoCapture(0)
        ret, _ = camera.read()
    finally:
        if not ret:
            camera.release()
            for index in range(2):
                sleep(0.25)
                try:
                    camera = cv2.VideoCapture(index)
                    ret, _ = camera.read()
                    if ret: break
                except: camera.release(); continue
            if not ret: camera.release(); raise Exception("Nie znaleziono kamery")

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

    try:
        while running:
            verdict = 0
            x = None
            for i in range(5):
                image, frame = read_cam(camera)
                    
                prediction_hand = hand_model._model.predict(image, verbose=0)

                verdict += float(prediction_hand[0][0])

                # print([[hand_labels[i], round(float(pred), 3)] for i, pred in enumerate(prediction_hand[0])])

                cv2.imshow("Kamera", frame)
                    
                if cv2.waitKey(1) == 27:
                    running = False
                    break

            verdict /= 5
    
            x, y, trash_class_idx = None

            if verdict < 0.15 and running:
                image, frame = read_cam(camera)

                prediction_trash_class = waste_model._model.predict(image, verbose=0)[0]

                if prediction_trash_class != 0: # 0 = neutralna klasa
                    prediction_location_x = np.array(location_x_model._model.predict(image, verbose=0)[0]).tolist()
                    prediction_location_y = np.array(location_y_model._model.predict(image, verbose=0)[0]).tolist()

                    trash_class_idx = np.argmax(prediction_trash_class)

                    x = [prediction_location_x.index(el) for el in prediction_location_x if el > 0.2]
                    y = [prediction_location_y.index(el) for el in prediction_location_y if el > 0.2]

            else:
                stop = True

            if (x and y and trash_class_idx) or stop: packet = construct_packet((x, y), trash_class_idx, stop)
            if packet: send_packet(packet)

    finally:
        camera.release()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Exception: {e}")
    raise