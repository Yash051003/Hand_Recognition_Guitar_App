from django.shortcuts import render

def landing(request):
    return render(request, 'streamapp/landing.html')

def index(request):
    return render(request, 'streamapp/index.html')