import csv
from dataclasses import dataclass

from openai import OpenAI
import torch
from django import forms
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, JsonResponse
from django.urls import path
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt

from .audio_model import combined_model

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
    openai_client = OpenAI()

    print(f"Piece 1: {piece_1.name} ({piece_1.file.size / 1000}kb)")
    print(f"Piece 2: {piece_2.name} ({piece_2.file.size / 1000}kb)")

    spec_size = 1500
    dim = 64
    drop = 0.3
    in_channels_class = 128
    model = combined_model.combined_model(dim, drop, in_channels_class, spec_size)
    model.load_state_dict(torch.load("./feedback/audio_model/checkpoints/model_epoch_13.pt"))

    test_feedback = get_feedback_from_file()

    print("Finished loading test feedback")
    
    print(test_feedback[0])
    print(test_feedback[1])

    response = openai_client.responses.create(
        model="gpt-5-nano",
        instructions="I want you to read feedback given to you for two pieces of music and provide 5 different words for each song that give a sense of the feedback. Don't just copy single words from the text unless they are musically relevant. I also want whether the feedback about the word was positive or negative. For example you may use the word expressive and positive. For each feedback given be consistent with your words you use to an extent but do not repeat all the words that you use. Give it to me in the form of: Word_positive/negative Word_positive/negative",
        input=f"Piece 1: ${test_feedback[0][8]}\n\nPiece 2: ${test_feedback[1][8]}"
    )

    print(response.output_text)

    return Feedback(50, response.output_text)

def get_feedback_from_file():
    with open("./feedback/audio_model/abrsm_lmth25.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader, None)

        return [
            next(reader, None),
            next(reader, None)
        ]


        # for row in reader:
        #     feedback = row[8]
        #     all_feedback += feedback

@csrf_exempt
def train_view(request):
    openai_client = OpenAI()

    with open("./feedback/audio_model/abrsm_lmth25.csv") as csv_file:
        with open("./feedback/audio_model/chatgpt_abrsm_lmth25.csv", "w") as chatgpt_file:
            reader = csv.reader(csv_file, delimiter=",")
            next(reader, None)

            writer = csv.writer(chatgpt_file, delimiter=",")
            writer.writerow(["performance_id", "piece_1_feedback", "piece_2_feedback"])

            for row in reader:
            # row = next(reader, None)
                performance_id = row[0]
                feedback = row[8]

                response = openai_client.responses.create(
                    model="gpt-5-nano",
                    instructions="I want you to read feedback given to you for two pieces of music and provide 5 different words for each song that give a sense of the feedback. Don't just copy single words from the text unless they are musically relevant. I also want whether the feedback about the word was positive or negative. For example you may use the word expressive and positive. For each feedback given be consistent with your words you use to an extent but do not repeat all the words that you use. The output you must use is this: 'word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative]\nword/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative]'.",
                    input=feedback
                )

                response_text = response.output_text.split("\n")

                piece_1_feedback = response_text[0].split(" ")
                piece_2_feedback = response_text[1].split(" ")

                writer.writerow([performance_id, piece_1_feedback, piece_2_feedback])

    return HttpResponse("Done!")
    
