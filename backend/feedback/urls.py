from django.urls import path

from feedback.views import feedback_view


urlpatterns = [
    path("", feedback_view)
]
