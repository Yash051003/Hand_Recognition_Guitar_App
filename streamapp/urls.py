# streamapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # ⬅️ This is what shows the HTML page
]
