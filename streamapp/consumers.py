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
import asyncio
import pygame
from pathlib import Path

# Initialize pygame for sound playback
pygame.mixer.init()
pygame.mixer.set_num_channels(16)  # Allow multiple sounds to play simultaneously

# Sound file paths - these will be relative to the Django project's BASE_DIR
CHORD_SOUNDS = {
    # Major Chords
    "A": "app_client/Chords/Major/A.mp3",
    "B": "app_client/Chords/Major/B.mp3",
    "C": "app_client/Chords/Major/C.mp3",
    "D": "app_client/Chords/Major/D.mp3",
    "E": "app_client/Chords/Major/E.mp3",
    "F": "app_client/Chords/Major/F.mp3",
    "G": "app_client/Chords/Major/G.mp3",
    
    # Minor Chords
    "Am": "app_client/Chords/Minor/Am.mp3",
    "Bm": "app_client/Chords/Minor/Bm.mp3",
    "Cm": "app_client/Chords/Minor/Cm.mp3",
    "Dm": "app_client/Chords/Minor/Dm.mp3",
    "Em": "app_client/Chords/Minor/Em.mp3",
    "Fm": "app_client/Chords/Minor/Fm.mp3",
    "Gm": "app_client/Chords/Minor/Gm.mp3",
    
    # Flat Chords
    "A♭": "app_client/Chords/Flat/A flat .mp3",
    "B♭": "app_client/Chords/Flat/B flat .mp3",
    "D♭": "app_client/Chords/Flat/D flat .mp3",
    "E♭": "app_client/Chords/Flat/E flat .mp3",
    "G♭": "app_client/Chords/Flat/G flat .mp3",
    
    # Sharp Chords
    "A#": "app_client/Chords/Sharp/A sharp .mp3",
    "C#": "app_client/Chords/Sharp/C sharp.mp3",
    "D#": "app_client/Chords/Sharp/D sharp.mp3",
    "F#": "app_client/Chords/Sharp/F sharp.mp3",
    "G#": "app_client/Chords/Sharp/G sharp .mp3",
    
    # # Silence
    # "silence": "app_client/Chords/1-second-of-silence.mp3"
}

# Function to play a sound
def play_chord_sound(chord_name):
    if chord_name in CHORD_SOUNDS:
        sound_path = os.path.join(settings.BASE_DIR, CHORD_SOUNDS[chord_name])
        if os.path.exists(sound_path):
            try:
                sound = pygame.mixer.Sound(sound_path)
                sound.play()
                print(f"Playing sound for chord: {chord_name}")
                return True
            except Exception as e:
                print(f"Error playing sound: {e}")
                return False
        else:
            print(f"Sound file not found: {sound_path}")
            return False
    else:
        print(f"No sound file defined for chord: {chord_name}")
        return False

# Add app_client directory to path using more robust path resolution
APP_CLIENT_DIR = Path(settings.BASE_DIR) / 'app_client'
if APP_CLIENT_DIR.exists():
    sys.path.append(str(APP_CLIENT_DIR))
    try:
        from keypoint_classifier import KeyPointClassifier
        print(f"✅ Successfully imported KeyPointClassifier from {APP_CLIENT_DIR}")
    except ImportError as e:
        print(f"❌ Failed to import KeyPointClassifier: {e}")
else:
    print(f"❌ app_client directory not found at {APP_CLIENT_DIR}")

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
            # Load label mapping with more robust path handling
            csv_path = Path(settings.BASE_DIR) / 'app_client' / 'keypoint_classifier_label.csv'
            
            if not csv_path.exists():
                print(f"❌ CSV file not found at {csv_path}")
                # Try to find it in alternative locations
                alt_csv_paths = [
                    Path(settings.BASE_DIR) / 'app_client' / 'model' / 'keypoint_classifier' / 'keypoint_classifier_label.csv',
                    Path(settings.BASE_DIR) / 'model' / 'keypoint_classifier' / 'keypoint_classifier_label.csv'
                ]
                
                for alt_path in alt_csv_paths:
                    if alt_path.exists():
                        csv_path = alt_path
                        print(f"✅ Found CSV file at alternative location: {csv_path}")
                        break
            
            if csv_path.exists():
                with open(csv_path, encoding='utf-8-sig') as f:
                    self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
                print(f"✅ Successfully loaded labels from {csv_path}")
            else:
                # Fallback to hardcoded labels if file not found
                self.keypoint_classifier_labels = [
                    "Fist", "index", "thumb", "thumbindex", "thumbindexmiddle", 
                    "thumbindexmiddleringpinky", "indexpinky", "Not Applicable"
                ]
                print("⚠️ Using hardcoded fallback labels")
        except Exception as e:
            print(f"❌ Error loading KeyPointClassifier: {e}")
            # Fallback to TFLite model
            self.initialize_tflite_model()

        # Store current hand gestures
        self.current_gestures = {"Left": None, "Right": None}
        
        # Add a cooldown timer to prevent rapid-fire chord detection
        self.last_chord_time = 0 #seconds
        self.chord_cooldown = 0.9 #seconds

    def initialize_tflite_model(self):
        try:
            # Try to load TFLite model as fallback
            model_path = Path(settings.BASE_DIR) / 'app_client' / 'model.tflite'
            
            if not model_path.exists():
                print(f"❌ TFLite model not found at {model_path}")
                # Try alternative locations
                alt_model_path = Path(settings.BASE_DIR) / 'model.tflite'
                if alt_model_path.exists():
                    model_path = alt_model_path
                    print(f"✅ Found TFLite model at alternative location: {model_path}")
            
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✅ TFLite model loaded from {model_path}")
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
                    
                    # Play sound if chord detected
                    if chord:
                        # Run in a separate thread to avoid blocking
                        loop = asyncio.get_event_loop()
                        loop.run_in_executor(None, play_chord_sound, chord)
                        
                        # Only send prediction when chord is actually detected
                        await self.send(text_data=json.dumps({
                            "prediction": chord
                        }))
                        
                        print(f"Current gestures: {self.current_gestures}, Identified chord: {chord}")
                
                # Don't send "No chord detected" message
                # else:
                #     await self.send(text_data=json.dumps({
                #         "prediction": "Waiting for hand gestures..."
                #     }))
            
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
                            
                            # Play sound if chord detected
                            if chord:
                                # Run in a separate thread to avoid blocking
                                loop = asyncio.get_event_loop()
                                loop.run_in_executor(None, play_chord_sound, chord)
                                
                                # Only send message when chord is detected
                                
                                await self.send(text_data=json.dumps({
                                    "prediction": chord
                                    
                                }))
                            
                        # Don't send messages when no hands are detected
                        # else:
                        #     await self.send(text_data=json.dumps({
                        #         "prediction": "No hands detected"
                        #     }))
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
        import time
        current_time = time.time()
        
        left_gesture = self.current_gestures.get("Left")
        right_gesture = self.current_gestures.get("Right")
        
        # If we have both hand gestures, look up the chord
        if left_gesture and right_gesture:
            # Check cooldown
            if current_time - self.last_chord_time < self.chord_cooldown:
                return None
                
            chord = CHORD_MAPPINGS.get((left_gesture, right_gesture))
            
            # Reset after identifying a chord
            if chord:
                self.last_chord_time = current_time
                self.current_gestures = {"Left": None, "Right": None}
                
            return chord
            
        return None

