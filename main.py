import cv2
import time
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


from returnquery import query

import pygame, pygame.font, pygame.event, pygame.draw, string
from pygame.locals import *


def display_box(screen, message):
    fontobject=pygame.font.SysFont('Arial', 22)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                (10, (screen.get_height() / 2) - 10))
    pygame.display.flip()


def showText(text):
        # Graphics initialization
    full_screen = False    
    window_size = (1024, 200)
    pygame.init()      
    if full_screen:
        surf = pygame.display.set_mode(window_size, HWSURFACE | FULLSCREEN | DOUBLEBUF)
    else:
        surf = pygame.display.set_mode(window_size)

    # Create a display box
    display_box(surf, text)
    time.sleep(5)
    pygame.display.quit()

    pygame.quit()

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
        self.model = None

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
        high = -1
        for key, value in links.items():
            phrase, index = value
            if index > high:
                high = index

        return high

    def create_phrase_folder(self, phrase):
        # link folder name with phrase in json
        phrase_folder = phrase.replace(" ", "-")

        try:
            os.mkdir(self.DATA_PATH)
        except:
            pass

        link = {}

        try:
            with open(self.link_file_path, "r") as f:
                link = json.loads(f.read())
        except:
            pass

        index = self.find_last_index(link)

        if index == -1:
            index = 0
        else:
            index += 1

        link[phrase_folder] = [phrase, index]
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


    def load_data(self):
        self.label_map = {}
        self.phrase_link = {}

        self.actions = []

        with open(self.link_file_path, "r") as f:
            links = json.loads(f.read())

            for phrase_folder, values in links.items():
                self.label_map[values[0]] = values[1]
                self.actions.append(values[0])
                self.phrase_link[values[0]] = phrase_folder

        self.actions = np.array(self.actions)

    def pre_process(self):
        self.load_data()

        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(
                        os.path.join(
                            self.DATA_PATH, self.phrase_link[action], str(sequence), f"{frame_num}.npy"
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
            LSTM(64, return_sequences=True, activation="tanh", input_shape=(30, 1662))
        )
        model.add(LSTM(128, return_sequences=True, activation="tanh"))
        model.add(LSTM(64, return_sequences=False, activation="tanh"))
        model.add(Dense(64, activation="tanh"))
        model.add(Dense(32, activation="tanh"))
        model.add(Dense(16, activation="tanh"))
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
        try:
            model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
        except KeyboardInterrupt as e:
            pass

        print(model.summary())

        model.save("action.h5")
        self.model = model

        self.show_accuracy(model, X_test, y_test)

    def load_model(self):
        self.load_data()
        model = self.get_model()

        model.load_weights("action.h5")

        self.model = model

    def show_accuracy(self, model, X_test, y_test):
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        multilabel_confusion_matrix(ytrue, yhat)
        print("Accuracy Score: ")
        print(accuracy_score(ytrue, yhat))


    def predict(self):

        if (self.model == None):
            print("Load the model first")
            return

        self.load_data()

        cap = cv2.VideoCapture(0)

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.8

        s_time = 0

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)  as holistic:
            while cap.isOpened():

                ret, frame = cap.read()
                
                # make detection
                image, results = self.mediapipe_detection(frame, holistic)
                

                self.draw_landmarks(image, results)
                
                
                # prediction login
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-self.sequence_length:]

                text = ""
                
                if len(sequence) == 30:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    # print(actions[np.argmax(res)])
                
                
                #3. Viz logic
                    if np.unique(predictions[-15:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold: 
                            text = self.actions[np.argmax(res)]
                            if len(sentence) > 0: 
                                if self.actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(self.actions[np.argmax(res)])
                                    if (self.actions[np.argmax(res)] != "-"):
                                        s_time = time.time()
                            else:
                                sentence.append(self.actions[np.argmax(res)])

                    print(predictions[-15:])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]


                    if (s_time != 0):

                        print(time.time() - s_time)

                        if (time.time() - s_time > 5):

                            if len(sentence) > 0:
                                q = " ".join(sentence).replace("-", "")

                                if (q.strip() != ""):
                                    
                                    cv2.putText(
                                        image,
                                        "Searching....",
                                        (120, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 255, 0),
                                        4,
                                        cv2.LINE_AA,
                                    )
                                    cv2.imshow(self.window_name, image)
                                    cv2.waitKey(10)

                                    try:
                                        
                                        response = query(q)
                                    except Exception:
                                        response = "I couldn't find any results."
                                    sentence = []

                                    print(response)
                                    showText(response)
                                    s_time = 0
                                    sequence = []
                                    predictions = []
                        
                            
                    # Viz probabilities
                    # image = prob_viz(res, actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, " ".join(sentence).replace("-", ""), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                cv2.imshow(self.window_name, image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def display(self):
        print("-" * 30)
        print("info")
        print(f"Model loaded = {self.model != None}")
        print("-" * 30)
        print("1. Add a phrase")
        print("2. train")
        print("3. load model")
        print("4. predict")
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
                self.load_model()

            elif int(option) == 4:
                self.predict()

            elif int(option) == 5:
                break


asl = Asl()
asl.run()
