from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http.response import StreamingHttpResponse
from .aiexcersice import *
from .camera import *


# Create your views here.

def intropage(request):
    return render(request,'intro.html')

def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('ai_index')
        else:
            return HttpResponse("Username or Password is incorrect!!!")

    return render(request, 'login.html')

def LogoutPage(request):
    global b
    del b
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def ai_index(request):
    return render(request,'index.html')
@login_required(login_url='login')
def ai_pushup(request):
    global b
    b.reset()
    return render(request, 'pushup.html')
@login_required(login_url='login')
def ai_squats(request):
    global b
    b.reset()
    return render(request, 'squats.html')
@login_required(login_url='login')
def ai_situp(request):
    global b
    b.reset()
    return render(request, 'situp.html')
@login_required(login_url='login')
def ai_biceps(request):
    global b
    b.reset()
    return render(request, 'biceps.html')
@login_required(login_url='login')
def ai_result(request):
    return render(request, 'result.html')


def index_gen(camera):
    while True:
        frame = camera.intro()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def index_feed(request):
    global b
    b = VideoCamera()
    return StreamingHttpResponse(index_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')

def pushup_gen(camera):
    while True:
        frame = camera.pushup()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def pushup_feed(request):
    return StreamingHttpResponse(pushup_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')

def squats_gen(camera):
    while True:
        frame = camera.squats()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def squats_feed(request):
    return StreamingHttpResponse(squats_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')

def situp_gen(camera):
    while True:
        frame = camera.situp()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def situp_feed(request):
    return StreamingHttpResponse(situp_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')

def biceps_gen(camera):
    while True:
        frame = camera.biceps()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def biceps_feed(request):
    return StreamingHttpResponse(biceps_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')

def result_gen(camera):
    while True:
        frame = camera.result()
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def result_feed(request):
    return StreamingHttpResponse(result_gen(b),
					content_type='multipart/x-mixed-replace; boundary=frame')


