import cv2
import os
import mediapipe as mp
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


class Asl:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.no_sequences = 30
        self.sequence_length = 30
        self.DATA_PATH = os.path.join("MP_DATA")
        self.link_file_path = os.path.join(self.DATA_PATH, "link.json")
        self.window_name = "ASL"
        self.label_map = {}
        self.actions = []

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 256, 121), thickness=1, circle_radius=1
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=2, circle_radius=2
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(121, 44, 250), thickness=2, circle_radius=2
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    def extract_keypoints(self, results):
        pose = (
            np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in results.pose_landmarks.landmark
                ]
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 4)
        )
        face = (
            np.array(
                [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
            ).flatten()
            if results.face_landmarks
            else np.zeros(468 * 3)
        )
        lh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )
        return np.concatenate([pose, face, lh, rh])

    def find_last_index(self, links):
        high = 0
        for key, value in links.items():
            phrase, index = value
            if index > high:
                high = index

        return high

    def create_phrase_folder(self, phrase):
        # link folder name with phrase in json
        phrase_folder = phrase.replace(" ", "-")

        link = {}

        try:
            with open(self.link_file_path, "r") as f:
                link = json.loads(f.read())
        except:
            pass

        index = self.find_last_index(link)

        link[phrase_folder] = [phrase, index + 1]
        json_string = json.dumps(link)

        with open(self.link_file_path, "w") as f:
            f.write(json_string)

        # make sequence folder
        for seq_num in range(self.no_sequences):
            try:
                os.makedirs(os.path.join(self.DATA_PATH, phrase_folder, str(seq_num)))
            except:
                pass

        return phrase_folder

    def record(self):

        phrase = input("Enter the phrase: ")

        folder = self.create_phrase_folder(phrase)

        cap = cv2.VideoCapture(0)

        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:

            ret, frame = cap.read()

            width, height = frame.shape[:2]

            image = np.zeros((width, height))

            cv2.putText(
                frame,
                "Press s to Start",
                (120, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )

            cv2.imshow(self.window_name, frame)

            while True:
                key = cv2.waitKey(0)

                if key == ord("s"):
                    break
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()

            timer = 2
            while True:
                ret, frame = cap.read()
                cv2.putText(
                    frame,
                    f"Starting in {timer}  seconds.",
                    (120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, frame)
                timer -= 1
                cv2.waitKey(1000)
                if timer == 0:
                    break


            for sequence in range(self.no_sequences):
                for frame_num in range(self.sequence_length):

                    ret, frame = cap.read()

                    # make detection
                    image, results = self.mediapipe_detection(frame, holistic)

                    self.draw_landmarks(image, results)

                    cv2.putText(
                        image,
                        f"Collecting frames for {folder} Video Number {sequence}. CountDown: {self.sequence_length - frame_num}",
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(self.window_name, image)

                    keypoints = self.extract_keypoints(results)

                    npy_path = os.path.join(
                        self.DATA_PATH, folder, str(sequence), str(frame_num)
                    )
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                timer = 2
                while True:
                    ret, image = cap.read()
                    cv2.putText(
                        image,
                        f"Next Frame in {timer} second",
                        (120, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(self.window_name, image)
                    cv2.waitKey(1000)
                    timer -= 1

                    if timer == 0:
                        break

            cap.release()
            cv2.destroyAllWindows()

    def pre_process(self):
        self.label_map = {}

        self.actions = []

        with open(self.link_file_path, "r") as f:
            links = json.loads(f.read())

            for phrase_folder, values in links.items():
                self.label_map[values[1]] = values[0]
                self.actions.append(values[0])

        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(
                        os.path.join(
                            self.DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                        )
                    )
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])

        X = np.array(sequences)

        y = to_categorical(labels).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        return X_train, X_test, y_train, y_test

    def get_model(self):

        model = Sequential()
        model.add(
            LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662))
        )
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.actions.shape[0], activation="softmax"))
        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        return model

    def train(self):

        X_train, X_test, y_train, y_test = self.pre_process()

        log_dir = os.path.join("Logs")
        tb_callback = TensorBoard(log_dir=log_dir)

        model = self.get_model()
        model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

        print(model.summary())

        model.save("action.h5")

    def load_model(self):
        model = self.get_model()

        model.load_weights("action.h5")

        return model

    def show_accuracy(self, model, X_test, y_test):
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        multilabel_confusion_matrix(ytrue, yhat)
        print(accuracy_score(ytrue, yhat))

    def display(self):
        print("1. Add a phrase")
        print("2. train")
        print("-" * 30)

    def run(self):
        while True:
            self.display()

            option = input("Choose: ")

            if int(option) == 1:
                # phrase = input("Enter your phrase: ")
                self.record()

            elif int(option) == 2:
                self.train()

            elif int(option) == 3:
                break


asl = Asl()
asl.run()
