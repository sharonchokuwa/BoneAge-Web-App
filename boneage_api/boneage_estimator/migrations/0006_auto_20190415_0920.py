# Generated by Django 2.2 on 2019-04-15 01:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('boneage_estimator', '0005_auto_20190414_1016'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='prediction',
            field=models.IntegerField(null=True),
        ),
    ]
