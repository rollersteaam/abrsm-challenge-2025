from django.urls import path

from feedback.views import feedback_view, train_view


urlpatterns = [
    path("", feedback_view),
    path("train/", train_view)
]
