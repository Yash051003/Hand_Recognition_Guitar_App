#!/usr/bin/env python
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set the Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mlsite.settings")

# Import and apply the WSGI middleware
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application() 