#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        # Use direct path to the model file in the current directory if not specified
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'keypoint_classifier.tflite')
            
        # Check if the file exists, try alternative paths if needed
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, trying alternative locations...")
            alternatives = [
                os.path.join(base_dir, 'model.tflite'),
                os.path.join(base_dir, 'model/keypoint_classifier/keypoint_classifier.tflite')
            ]
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Found model at {model_path}")
                    break
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                                  num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
