"""boneage_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.conf.urls import url
from boneage_estimator.views import estimate, post_new, post_detail

from django.conf import settings
from django.views.static import serve


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^boneage_estimation/estimate/$', estimate, name='estimate'),
    url(r'^boneage_estimation/post_new/$', post_new, name='post_new'),
    url(r'^boneage_estimation/post_detail/$', post_detail, name='post_detail'),
    
]

if settings.DEBUG:
	urlpatterns += [
	url(r'^media/(?P<path>.*)/$',
		serve, {'document_root':
		settings.MEDIA_ROOT,}),
	]
