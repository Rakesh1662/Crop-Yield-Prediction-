from django.contrib import admin
from django.urls import path
from fakenews import views
from django.conf.urls import url
from .import views
urlpatterns = [
    url('admin/', admin.site.urls),
    path('index/',views.loginuser),
]