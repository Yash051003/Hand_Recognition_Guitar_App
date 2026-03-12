from django.urls import re_path
from streamapp import consumers

websocket_urlpatterns = [
    re_path(r'ws/stream/$', consumers.StreamConsumer.as_asgi()),
]