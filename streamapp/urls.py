# streamapp/urls.py
from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', views.landing, name='landing'),   # landing page at /
    path('app/', views.index, name='app'),      # guitar app at /app/
]
