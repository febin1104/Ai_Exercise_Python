"""registration URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app1 import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.intropage,name='intro'),
    path('signup/',views.SignupPage,name='signup'),
    path('login/',views.LoginPage,name='login'),
    path('logout/',views.LogoutPage,name='logout'),

    path('ai_index/',views.ai_index, name='ai_index'),
    path('index_feed',views.index_feed, name='index_feed'),

    path('ai_pushup/',views.ai_pushup, name='ai_pushup'),
    path('pushup_feed',views.pushup_feed, name='pushup_feed'),

    path('ai_squats/',views.ai_squats, name='ai_squats'),
    path('squats_feed',views.squats_feed, name='squats_feed'),

    path('ai_situp/',views.ai_situp, name='ai_situp'),
    path('situp_feed',views.situp_feed, name='situp_feed'),

    path('ai_biceps/',views.ai_biceps, name='ai_biceps'),
    path('biceps_feed',views.biceps_feed, name='biceps_feed'),

    path('ai_result/',views.ai_result, name='ai_result'),
    path('result_feed',views.result_feed, name='result_feed'),



]