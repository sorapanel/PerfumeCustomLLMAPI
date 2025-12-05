from django.urls import path
from .views import CustomProcessingView

urlpatterns = [
    path('response/', CustomProcessingView.as_view(), name='respense'),
]