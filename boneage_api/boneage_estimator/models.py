from django.db import models
from django.conf import settings

# Create your models here.
class Post(models.Model):
    userId = models.IntegerField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=6)
    image = models.ImageField(blank=False,
             upload_to="uploads/")
    carpal_image =  models.ImageField(blank=True, 
    	null= True, upload_to="processed/")
    prediction = models.IntegerField(null = True)
   
 