import csv
from dataclasses import dataclass
from typing import List

import numpy as np
from openai import OpenAI
import torch
from django import forms
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, JsonResponse
from django.urls import path
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics.pairwise import cosine_similarity

from .audio_model import combined_model

@dataclass
class Piece():
    # name: str
    file: File


@dataclass
class Feedback():
    score: int
    feedback: str


class FeedbackForm(forms.Form):
    piece_1_audio = forms.FileField()
    # piece_1_name = forms.CharField()
    piece_2_audio = forms.FileField()
    # piece_2_name = forms.CharField()


@csrf_exempt
def feedback_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Request not POST")

    form = FeedbackForm(request.POST, request.FILES)

    if not form.is_valid():
        return HttpResponseBadRequest("Form invalid")
    
    piece_1 = Piece(
        # form.cleaned_data["piece_1_name"],
        form.cleaned_data["piece_1_audio"]
    )
    piece_2 = Piece(
        # form.cleaned_data["piece_2_name"],
        form.cleaned_data["piece_2_audio"]
    )

    feedback = get_feedback_from_pieces(piece_1, piece_2)

    return JsonResponse(feedback.__dict__)


def get_feedback_from_pieces(piece_1: Piece, piece_2: Piece) -> Feedback:
    openai_client = OpenAI()

    print(f"Piece 1 ({piece_1.file.size / 1000}kb)")
    print(f"Piece 2 ({piece_2.file.size / 1000}kb)")

    # spec_size = 1500
    # dim = 64
    drop = 0.3
    # in_channels_class = 128
    model = combined_model.combined_model(drop)
    model.load_state_dict(torch.load("./feedback/audio_model/checkpoints/best_model.pt"))
    model.eval()

    embed_dict = np.load("./feedback/audio_model/emb_dict.npz")
    grade_dict = np.load("./feedback/audio_model/grade_dict.npz")

    word2vec_data = np.load("./feedback/audio_model/w2v_dict.npz")
    word2vec = { key: word2vec_data[key] for key in word2vec_data.files }

    def eval_piece(piece: Piece):
        file_name = piece.file.name.split(".mp3")[0]
        print(f"Evaluating piece ${file_name}")

        grade_embedding = torch.tensor(grade_dict[file_name]).unsqueeze(0)
        song_embedding = torch.tensor(embed_dict[file_name]).mean(axis=0).unsqueeze(0)
        final_embedding = torch.cat((song_embedding, grade_embedding), dim=1).float()
        print(final_embedding.shape)
        
        pred_mask, pred_word = model(final_embedding)

        # mark_prediction.squeeze().numpy()

        # mark_prediction = np.random.rand(40)
        mark_prediction = pred_mask.detach().squeeze().numpy()

        score = np.argmax(mark_prediction) + 60

        # word_prediction = np.random.rand(5, 768)
        word_prediction = pred_word.detach().squeeze().numpy()
        word_prediction = word_prediction + np.random.uniform(-5, 5, size=word_prediction.shape)

        best_score = np.array([500., 500., 500., 500., 500.])
        best_word = ["test_word", "test_word", "test_word", "test_word", "test_word"]

        for word, embedding in word2vec.items():
            embedding = embedding.reshape(1, -1)
            sims = (1 - cosine_similarity(word_prediction, embedding).squeeze()) / 2
            
            comparison = sims < best_score

            best_score[comparison] = sims[comparison]

            for i in range(len(comparison)):
                if comparison[i]:
                    best_word[i] = word

            if comparison.any():
                print("New best")
                print(best_word)
                print(best_score)
        
        print("Piece descriptions")
        
        piece_descriptions = " ".join(best_word)
        print(piece_descriptions)

        return score, piece_descriptions

    eval_piece_1 = eval_piece(piece_1)
    eval_piece_2 = eval_piece(piece_2)

    response = openai_client.responses.create(
        model="gpt-5-nano",
        instructions="Given a set of 5 words which summarise feedback given on a performance of a piece of music and whether the words correspond to positive or negative feedback. Write feedback in the style of these given examples: Burgm√ºller's Ballade was firmly grounded from the outset, and details of accents and articulation vividly brought out the character of the opening passages. There were moments of slight unevenness of tone and rhythm along the way, and there was scope for more nuanced balance at times, but this was largely an assertive account. Your playing of Fountain in the Rain evoked the musical style effectively, delicately defining the figures, all warmly supported by effective use of the sustaining pedal. The more florid cadenza-like passages in the middle were particularly compelling, though subsequently there needed more rhythmical regularity to be upheld. Otherwise this was stylish playing. You brought flair and poise to the presentation, and a real feeling of performance awareness was communicated. While rhythmical consistency wasn't fully assured, other elements of the technical delivery showed greater security and confidence, and there was expressive input achieved, especially in the rippling character of the second work. Congratulations on bringing such strong intensity and involvement to your playing! A lively tempo suited the style of the Neilsen and the tone was bright to open, with musical details observed, although more clarity in projection would have further animated the narratve. Coordination of the hands often needed to be tighter but a good sense of momentum was maintained. The Chopin Valse was mostly steady in pulse if initially on the reserved side tempo wise, needing further rhythmic lilt. Dynamic contrast and inparticular more shaping of phrases would have supported the elegance of the style further, but the notes were largely secure. Reliability of notes and flow were present in this performance with contrast between the styles apparent although needing further vibrancy in detail and more tonal shaping. A good sense of focus was present in your playing.",
        input=f"Piece 1: ${eval_piece_1[1]} Piece 2: ${eval_piece_2[1]}"
    )

    # print(response.output_text)

    return Feedback((int(eval_piece_1[0]) + int(eval_piece_2[0])) / 2, response.output_text)

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

    done = 1
    already_done = set()

    with open("./feedback/audio_model/chatgpt_abrsm_lmth25.csv") as chatgpt_file:
        reader = csv.reader(chatgpt_file, delimiter=",")

        for row in reader:
            performance_id = row[0]
            already_done.add(performance_id)

    total = 0
    with open("./feedback/audio_model/abrsm_lmth25.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            total += 1

    with open("./feedback/audio_model/abrsm_lmth25.csv") as csv_file:
        with open("./feedback/audio_model/chatgpt_abrsm_lmth25.csv", "a") as chatgpt_file:
            reader = csv.reader(csv_file, delimiter=",")
            next(reader, None)

            writer = csv.writer(chatgpt_file, delimiter=",")
            # writer.writerow(["performance_id", "piece_1_feedback", "piece_2_feedback"])

            # row = next(reader, None)
            for row in reader:
                performance_id = row[0]
                feedback = row[8]

                if performance_id in already_done:
                    continue

                response = openai_client.responses.create(
                    model="gpt-5-nano",
                    instructions="I want you to read feedback given to you for two pieces of music and provide 5 different words for each song that give a sense of the feedback. Don't just copy single words from the text unless they are musically relevant. I also want whether the feedback about the word was positive or negative. For example you may use the word expressive and positive. For each feedback given be consistent with your words you use to an extent but do not repeat all the words that you use. The output you must use is this: 'word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative]\nword/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative] word/[positive/negative]'.",
                    input=feedback
                )

                response_text = response.output_text.split("\n")

                piece_1_feedback = response_text[0].split(" ")
                piece_2_feedback = response_text[1].split(" ")

                writer.writerow([performance_id, piece_1_feedback, piece_2_feedback])
                chatgpt_file.flush()
                print(f"{[performance_id, piece_1_feedback, piece_2_feedback]} ({done}/{total - len(already_done)})")
                done += 1

    return HttpResponse("Done!")

def clean_feedback(feedback: str) -> List[str]:
    def strip_func(char: str) -> str:
        return char.replace(
            "[", ""
        ).replace(
            "]", ""
        ).replace(
            "'", ""
        ).replace(
            "\"", ""
        ).replace(
            " ", ""
        ).lower()

    return list(map(strip_func, feedback.split(",")))

@csrf_exempt
def clean_view(request):
    with open("./feedback/audio_model/chatgpt_abrsm_lmth25_cleaned.csv", "w") as new_file:
        with open("./feedback/audio_model/chatgpt_abrsm_lmth25.csv") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader, None)

            writer = csv.writer(new_file, delimiter="|")
            writer.writerow(["performance_id", "piece_1_feedback", "piece_2_feedback"])

            already_done = set()

            for row in reader:
                performance_id = row[0]

                # De-duplicate results
                if performance_id in already_done:
                    continue

                already_done.add(performance_id)

                # Remove extra characters from chatgpts response
                piece_1_feedback = clean_feedback(row[1])
                piece_2_feedback = clean_feedback(row[2])

                writer.writerow([performance_id, piece_1_feedback, piece_2_feedback])
    
    return HttpResponse("Data cleaned")
