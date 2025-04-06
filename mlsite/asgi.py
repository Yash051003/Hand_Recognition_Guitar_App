import os
from django.core.asgi import get_asgi_application

# Set Django settings module before any other imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlsite.settings')

# Initialize Django ASGI application first
django_asgi_app = get_asgi_application()

# Import other components after Django is initialized
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import streamapp.routing  # Import app's routing after Django is initialized

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            streamapp.routing.websocket_urlpatterns
        )
    ),
})
