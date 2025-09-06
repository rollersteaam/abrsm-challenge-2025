from dataclasses import dataclass

from django import forms
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, JsonResponse
from django.urls import path
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt


@dataclass
class Piece():
    name: str
    file: File


@dataclass
class Feedback():
    score: int
    feedback: str


class FeedbackForm(forms.Form):
    piece_1_audio = forms.FileField()
    piece_1_name = forms.CharField()
    piece_2_audio = forms.FileField()
    piece_2_name = forms.CharField()


@csrf_exempt
def feedback_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Request not POST")

    form = FeedbackForm(request.POST, request.FILES)

    if not form.is_valid():
        return HttpResponseBadRequest("Form invalid")
    
    piece_1 = Piece(
        form.cleaned_data["piece_1_name"],
        form.cleaned_data["piece_1_audio"]
    )
    piece_2 = Piece(
        form.cleaned_data["piece_2_name"],
        form.cleaned_data["piece_2_audio"]
    )

    feedback = get_feedback_from_pieces(piece_1, piece_2)

    return JsonResponse(feedback.__dict__)


def get_feedback_from_pieces(piece_1: Piece, piece_2: Piece) -> Feedback:
    print(f"Piece 1: {piece_1.name} ({piece_1.file.size / 1000}kb)")
    print(f"Piece 2: {piece_2.name} ({piece_2.file.size / 1000}kb)")

    return Feedback(50, "No feedback")
