"""fakenewsdetect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
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
from fakenews import views
from django.conf.urls import url
from django.urls import include
urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^home', views.home, name='home'),
	url(r'^nvb', views.nvb, name='nvb'),
	url(r'^pac', views.pac, name='pac'),
	url(r'^svm', views.svm, name='svm'),
	url(r'^accuracy', views.accuracy, name='accuracy'),
	url(r'^loginuser', views.loginuser, name='loginuser'),
	url(r'^rf', views.rf, name='rf'),
	url(r'^input', views.input, name='input'),
	url(r'^test', views.test, name='test'),
	url(r'^simple_upload', views.simple_upload, name='simple_upload'),
	url(r'^simple', views.simple, name='simple'),
	url(r'^fileshow', views.fileshow, name='fileshow'),
	url(r'^fileshow1', views.fileshow1, name='fileshow1'),
	url(r'^$', views.loginpage, name='loginpage'),
	url(r'^register', views.register, name='register'),
	url(r'^reg', views.reg, name='reg'),
	url(r'^svr', views.svr, name='svr'),
	url(r'^fertilizerRf', views.fertilizerRf, name='fertilizerRf'),
	url(r'^fertilizerform', views.fertilizerform, name='fertilizerform'),
	path('',include('fakenews.urls')),
	
]
