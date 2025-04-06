import base64
import cv2
import numpy as np
import os
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
import tensorflow as tf
import mediapipe as mp
import sys
import csv
import itertools
import copy

# Add app_client directory to path to use the existing model classes
sys.path.append(os.path.join(settings.BASE_DIR, 'app_client'))
try:
    from model import KeyPointClassifier
except ImportError:
    # Fallback if import fails
    print("Failed to import KeyPointClassifier from app_client")

# Define chord mappings based on the gesture combinations
CHORD_MAPPINGS = {
    # Format: ("Left_hand_gesture", "Right_hand_gesture"): "Chord"
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
    ("Fist", "thumb"): "A♭",
    ("Fist", "thumbindex"): "B♭",
    ("Fist", "thumbindexmiddle"): "D♭",
    ("Fist", "thumbindexmiddleringpinky"): "E♭",
    ("Fist", "Fist"): "G♭",
    ("index", "thumb"): "A#",
    ("index", "thumbindex"): "C#",
    ("index", "thumbindexmiddle"): "D#",
    ("index", "thumbindexmiddleringpinky"): "F#",
    ("index", "Fist"): "G#",
}

class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("✅ WebSocket connected")

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Load KeyPointClassifier
        try:
            self.keypoint_classifier = KeyPointClassifier()
            # Load label mapping
            csv_path = os.path.join(settings.BASE_DIR, 'app_client', 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
            with open(csv_path, encoding='utf-8-sig') as f:
                self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
            print("✅ Successfully loaded KeyPointClassifier")
        except Exception as e:
            print(f"❌ Error loading KeyPointClassifier: {e}")
            # Fallback to TFLite model
            self.initialize_tflite_model()

        # Store current hand gestures
        self.current_gestures = {"Left": None, "Right": None}

    def initialize_tflite_model(self):
        try:
            # Try to load TFLite model as fallback
            model_path = os.path.join(settings.BASE_DIR, 'app_client', 'model.tflite')
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("✅ TFLite model loaded as fallback")
        except Exception as e:
            print(f"❌ Failed to load TFLite model: {e}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            
            # Check if received data contains hand landmarks
            if "hand_data" in data:
                hand_data = data.get("hand_data", [])
                
                # Process hand gesture data
                if hand_data:
                    for hand_info in hand_data:
                        hand_label = hand_info.get("hand")  # Left or Right
                        gesture = hand_info.get("gesture")
                        
                        # Store gesture for this hand
                        if hand_label in self.current_gestures:
                            self.current_gestures[hand_label] = gesture
                    
                    # Check if we have both hands detected for a chord
                    chord = self.identify_chord()
                    
                    # Send back the identified chord
                    await self.send(text_data=json.dumps({
                        "prediction": chord or "No chord detected"
                    }))
                    
                    print(f"Current gestures: {self.current_gestures}, Identified chord: {chord}")
                else:
                    await self.send(text_data=json.dumps({
                        "prediction": "Waiting for hand gestures..."
                    }))
            
            # Process raw frame data if no hand data is provided
            elif "frame" in data and "frame_data" not in data:
                # Extract and process frame
                frame_data = data.get("frame", "")
                
                if frame_data:
                    try:
                        # Decode base64 image
                        if ',' in frame_data:
                            frame_data = frame_data.split(',')[1]
                        image_bytes = base64.b64decode(frame_data)
                        np_array = np.frombuffer(image_bytes, np.uint8)
                        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                        
                        # Process with MediaPipe
                        results = self.process_frame(frame)
                        
                        # If hands detected, classify gestures
                        if results.multi_hand_landmarks:
                            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                                 results.multi_handedness):
                                hand_label = handedness.classification[0].label
                                
                                # Process landmarks to get features
                                landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                                
                                # Classify hand gesture using KeyPointClassifier
                                gesture_id = self.keypoint_classifier(pre_processed_landmark_list)
                                
                                # Update current gestures
                                if hand_label in self.current_gestures:
                                    self.current_gestures[hand_label] = self.keypoint_classifier_labels[gesture_id]
                            
                            # Identify chord
                            chord = self.identify_chord()
                            
                            await self.send(text_data=json.dumps({
                                "prediction": chord or "No chord detected"
                            }))
                        else:
                            await self.send(text_data=json.dumps({
                                "prediction": "No hands detected"
                            }))
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        await self.send(text_data=json.dumps({
                            "error": f"Processing failed: {str(e)}"
                        }))

        except Exception as e:
            print(f"Error in receive: {str(e)}")
            await self.send(text_data=json.dumps({
                "error": f"Processing failed: {str(e)}"
            }))

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        return results

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        # Extract keypoints
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
                
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
            
        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        def normalize_(n):
            return n / max_value
            
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        return temp_landmark_list

    def identify_chord(self):
        """Identify chord based on the current left and right hand gestures"""
        left_gesture = self.current_gestures.get("Left")
        right_gesture = self.current_gestures.get("Right")
        
        # If we have both hand gestures, look up the chord
        if left_gesture and right_gesture:
            chord = CHORD_MAPPINGS.get((left_gesture, right_gesture))
            
            # Reset after identifying a chord
            if chord:
                self.current_gestures = {"Left": None, "Right": None}
                
            return chord
            
        return None

