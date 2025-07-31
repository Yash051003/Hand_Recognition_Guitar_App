# consumers.py - Fixed Version

import base64
import cv2
import numpy as np
import os
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
import mediapipe as mp
import sys
import csv
import itertools
import copy
import asyncio
from pathlib import Path

# Try to import TensorFlow first, then fall back to TensorFlow Lite
try:
    import tensorflow as tf
    # Check if TensorFlow Lite is available
    if hasattr(tf, 'lite'):
        print("✅ TensorFlow with TFLite support loaded")
    else:
        print("⚠️ TensorFlow loaded but no TFLite support, trying tflite_runtime")
        import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        print("✅ TensorFlow Lite runtime loaded")
    except ImportError:
        print("❌ Neither TensorFlow nor TensorFlow Lite runtime found")
        tflite = None

# Add app_client directory to path
APP_CLIENT_DIR = Path(settings.BASE_DIR) / 'app_client'
if APP_CLIENT_DIR.exists():
    sys.path.append(str(APP_CLIENT_DIR))
    try:
        from keypoint_classifier import KeyPointClassifier
        print(f"✅ Successfully imported KeyPointClassifier from {APP_CLIENT_DIR}")
    except ImportError as e:
        print(f"❌ Failed to import KeyPointClassifier: {e}")
        KeyPointClassifier = None
else:
    print(f"❌ app_client directory not found at {APP_CLIENT_DIR}")
    KeyPointClassifier = None

# Define chord mappings
CHORD_MAPPINGS = {
    ("thumbindex", "Fist"): "A", ("thumb", "indexpinky"): "B",
    ("thumb", "thumbindexmiddle"): "C", ("thumb", "thumbindexmiddleringpinky"): "D",
    ("thumb", "Fist"): "E", ("thumb", "index"): "F", ("thumb", "thumbindex"): "G",
    ("thumbindex", "thumb"): "Am", ("thumbindex", "thumbindex"): "Bm",
    ("thumbindex", "thumbindexmiddle"): "Cm", ("thumbindex", "thumbindexmiddleringpinky"): "Dm",
    ("thumb", "thumb"): "Em", ("thumbindex", "index"): "Fm",
    ("thumbindex", "indexpinky"): "Gm", ("Fist", "thumb"): "A♭",
    ("Fist", "thumbindex"): "B♭", ("Fist", "thumbindexmiddle"): "D♭",
    ("Fist", "thumbindexmiddleringpinky"): "E♭", ("Fist", "Fist"): "G♭",
    ("index", "thumb"): "A#", ("index", "thumbindex"): "C#",
    ("index", "thumbindexmiddle"): "D#", ("index", "thumbindexmiddleringpinky"): "F#",
    ("index", "Fist"): "G#",
}

class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("✅ WebSocket connected")

        # Initialize MediaPipe Hands - FIXED: Use consistent attribute name
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=2,  # Changed from 4 to 2 since you only use Left/Right
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )

        # Initialize KeyPoint Classifier
        self.keypoint_classifier = None
        self.keypoint_classifier_labels = []
        
        if KeyPointClassifier is not None:
            try:
                self.keypoint_classifier = KeyPointClassifier()
                
                # Load the label file
                csv_path = Path(settings.BASE_DIR) / 'app_client' / 'keypoint_classifier_label.csv'
                
                if csv_path.exists():
                    with open(csv_path, encoding='utf-8-sig') as f:
                        self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
                    print(f"✅ Successfully loaded {len(self.keypoint_classifier_labels)} labels from {csv_path}")
                    print(f"Labels: {self.keypoint_classifier_labels}")
                else:
                    print(f"❌ CSV file not found at {csv_path}. Gesture recognition will fail.")
                    
            except Exception as e:
                print(f"❌ Error loading KeyPointClassifier: {e}")
                self.keypoint_classifier = None
                self.keypoint_classifier_labels = []
        else:
            print("❌ KeyPointClassifier not available")

        # Initialize gesture tracking
        self.current_gestures = {"Left": None, "Right": None}
        self.last_chord_time = 0 
        self.chord_cooldown = 0.9

    async def disconnect(self, close_code):
        """Clean up MediaPipe resources when disconnecting"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        print("🔌 WebSocket disconnected")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if "frame" in data:
                frame_data = data.get("frame", "").split(',')[1]
                image_bytes = base64.b64decode(frame_data)
                np_array = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("❌ Failed to decode frame")
                    return
                
                # Process the frame
                results = self.process_frame(frame)
                
                # Reset current gestures
                self.current_gestures = {"Left": None, "Right": None}
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Get hand label (Left or Right)
                        hand_label = handedness.classification[0].label
                        
                        # Calculate landmarks
                        landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                        pre_processed_list = self.pre_process_landmark(landmark_list)
                        
                        # Classify gesture
                        if self.keypoint_classifier and self.keypoint_classifier_labels:
                            try:
                                gesture_id = self.keypoint_classifier(pre_processed_list)
                                if 0 <= gesture_id < len(self.keypoint_classifier_labels):
                                    gesture_name = self.keypoint_classifier_labels[gesture_id]
                                    self.current_gestures[hand_label] = gesture_name
                                    print(f"🖐️ {hand_label} hand: {gesture_name}")
                                else:
                                    print(f"⚠️ Invalid gesture_id: {gesture_id}")
                            except Exception as e:
                                print(f"❌ Error in gesture classification: {e}")
                    
                    # Try to identify chord
                    chord = self.identify_chord()
                    if chord:
                        await self.send(text_data=json.dumps({"prediction": chord}))
                        print(f"🎵 Sent chord to client: {chord}")
                        
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            print(f"❌ Error in receive: {str(e)}")

    def process_frame(self, frame):
        """Process frame with MediaPipe Hands"""
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = self.hands.process(image)
            
            # Make image writable again
            image.flags.writeable = True
            
            return results
            
        except Exception as e:
            print(f"❌ Error in process_frame: {e}")
            return None

    def calc_landmark_list(self, image, landmarks):
        """Calculate landmark coordinates"""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        """Pre-process landmarks for classification"""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Normalize coordinates relative to wrist (first landmark)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        
        # Flatten the list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalize to [-1, 1] range
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value == 0: 
            return temp_landmark_list
            
        def normalize_(n): 
            return n / max_value
            
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        return temp_landmark_list

    def identify_chord(self):
        """Identify chord based on current gestures"""
        import time
        current_time = time.time()
        
        left_gesture = self.current_gestures.get("Left")
        right_gesture = self.current_gestures.get("Right")
        
        print(f"🎯 Current gestures - Left: {left_gesture}, Right: {right_gesture}")
        
        if left_gesture and right_gesture:
            # Check cooldown
            if current_time - self.last_chord_time > self.chord_cooldown:
                chord = CHORD_MAPPINGS.get((left_gesture, right_gesture))
                if chord:
                    self.last_chord_time = current_time
                    print(f"🎸 Chord found: {chord} from gestures ({left_gesture}, {right_gesture})")
                    return chord
                else:
                    print(f"❌ No chord mapping for gestures: ({left_gesture}, {right_gesture})")
            else:
                print(f"⏰ Chord in cooldown: {self.chord_cooldown - (current_time - self.last_chord_time):.1f}s remaining")
        
        return None