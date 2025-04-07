import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        img_data = base64.b64decode(data['image'])

        # Convert to numpy array and then to image
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ---- ML Processing Here ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Example: Add circle
        cv2.circle(gray, (100, 100), 30, (255, 0, 0), 5)
        result, buffer = cv2.imencode('.jpg', gray)

        processed_img = base64.b64encode(buffer).decode('utf-8')
        await self.send(text_data=json.dumps({'image': processed_img}))
