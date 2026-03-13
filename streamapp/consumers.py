import base64
import cv2
import numpy as np
import os
import json
import sys
import csv
import itertools
import copy
import time
from pathlib import Path

APP_CLIENT_DIR = Path(__file__).resolve().parent.parent / 'app_client'
sys.path.insert(0, str(APP_CLIENT_DIR))

from keypoint_classifier import KeyPointClassifier
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
import mediapipe as mp

try:
    import tensorflow as tf
    print("[OK] TensorFlow with TFLite support loaded")
except ImportError:
    print("[WARN] TensorFlow not found")

CHORD_MAPPINGS = {
    ("thumbindex", "Fist"): "A",
    ("thumb", "indexpinky"): "B",
    ("thumb", "thumbindexmiddle"): "C",
    ("thumb", "thumbindexmiddleringpinky"): "D",
    ("thumb", "Fist"): "E",
    ("thumb", "index"): "F",
    ("thumb", "thumbindex"): "G",
    ("thumbindex", "thumb"): "Am",
    ("thumbindex", "thumbindex"): "Bm",
    ("thumbindex", "thumbindexmiddle"): "Cm",
    ("thumbindex", "thumbindexmiddleringpinky"): "Dm",
    ("thumb", "thumb"): "Em",
    ("thumbindex", "index"): "Fm",
    ("thumbindex", "indexpinky"): "Gm",
    ("Fist", "thumb"): "A\u266d",
    ("Fist", "thumbindex"): "B\u266d",
    ("Fist", "thumbindexmiddle"): "D\u266d",
    ("Fist", "thumbindexmiddleringpinky"): "E\u266d",
    ("Fist", "Fist"): "G\u266d",
    ("index", "thumb"): "A#",
    ("index", "thumbindex"): "C#",
    ("index", "thumbindexmiddle"): "D#",
    ("index", "thumbindexmiddleringpinky"): "F#",
    ("index", "Fist"): "G#",
}


class StreamConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()
        print("[OK] WebSocket connected")

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.keypoint_classifier = None
        self.keypoint_classifier_labels = []

        try:
            self.keypoint_classifier = KeyPointClassifier()
            csv_path = Path(settings.BASE_DIR) / 'app_client' / 'keypoint_classifier_label.csv'
            if csv_path.exists():
                with open(csv_path, encoding='utf-8-sig') as f:
                    self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
                print(f"[OK] Loaded {len(self.keypoint_classifier_labels)} labels: {self.keypoint_classifier_labels}")
            else:
                print(f"[ERROR] CSV not found: {csv_path}")
        except Exception as e:
            print(f"[ERROR] KeyPointClassifier load failed: {e}")

        self.current_gestures = {"Left": None, "Right": None}
        self.last_chord_time = 0
        self.chord_cooldown = 0.9

    async def disconnect(self, close_code):
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        print("[OK] WebSocket disconnected")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if "frame" not in data:
                return

            frame_data = data.get("frame", "")
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]

            image_bytes = base64.b64decode(frame_data)
            np_array = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if frame is None:
                return

            results = self.process_frame(frame)
            self.current_gestures = {"Left": None, "Right": None}

            if results and results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    raw_label = handedness.classification[0].label
                    # Swap Left/Right - camera is mirrored so MediaPipe labels are flipped
                    hand_label = "Right" if raw_label == "Left" else "Left"

                    landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                    pre_processed_list = self.pre_process_landmark(landmark_list)

                    if self.keypoint_classifier and self.keypoint_classifier_labels:
                        try:
                            gesture_id = self.keypoint_classifier(pre_processed_list)
                            if 0 <= gesture_id < len(self.keypoint_classifier_labels):
                                gesture_name = self.keypoint_classifier_labels[gesture_id]
                                self.current_gestures[hand_label] = gesture_name
                                print(f"[HAND] {hand_label} hand: {gesture_name}")
                        except Exception as e:
                            print(f"[ERROR] Gesture classification: {e}")

                chord = self.identify_chord()
                if chord:
                    await self.send(text_data=json.dumps({"prediction": chord}))
                    print(f"[CHORD] Sent: {chord}")
                else:
                    await self.send(text_data=json.dumps({"prediction": "No chord detected"}))
            else:
                await self.send(text_data=json.dumps({"prediction": "No hands detected"}))

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode: {e}")
        except Exception as e:
            print(f"[ERROR] receive: {e}")

    def process_frame(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True
            return results
        except Exception as e:
            print(f"[ERROR] process_frame: {e}")
            return None

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for landmark in landmarks.landmark:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([x, y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for point in temp:
            point[0] -= base_x
            point[1] -= base_y
        flat = list(itertools.chain.from_iterable(temp))
        max_value = max(map(abs, flat))
        if max_value == 0:
            return flat
        return [n / max_value for n in flat]

    def identify_chord(self):
        current_time = time.time()
        left = self.current_gestures.get("Left")
        right = self.current_gestures.get("Right")
        print(f"[GESTURE] Left: {left}, Right: {right}")
        if left and right:
            if current_time - self.last_chord_time > self.chord_cooldown:
                chord = CHORD_MAPPINGS.get((left, right))
                if chord:
                    self.last_chord_time = current_time
                    print(f"[CHORD] {chord} from ({left}, {right})")
                    return chord
                else:
                    print(f"[MISS] No mapping for ({left}, {right})")
        return None