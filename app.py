import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from flask import Flask, render_template, request, send_file
import re
import heapq
import os

app = Flask(__name__)

# ---------------- Sentence Tokenizer ----------------

def custom_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence for sentence in sentences if sentence]


# ---------------- Frequency Based Summarization ----------------

def frequency_summary(text, num_sentences=3):

    sentences = custom_sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    stop_words = {
        "the", "is", "a", "an", "and", "of", "to",
        "in", "on", "for", "are", "was", "were",
        "this", "that", "it", "as", "by", "with"
    }

    word_frequency = {}

    for word in words:
        if word not in stop_words:
            word_frequency[word] = word_frequency.get(word, 0) + 1

    max_frequency = max(word_frequency.values())

    for word in word_frequency:
        word_frequency[word] = word_frequency[word] / max_frequency

    sentence_scores = {}

    for sentence in sentences:
        for word in re.findall(r'\b[a-zA-Z]+\b', sentence.lower()):
            if word in word_frequency:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequency[word]

    summary_sentences = heapq.nlargest(
        num_sentences,
        sentence_scores,
        key=sentence_scores.get
    )

    # Keep original order
    summary = [sentence for sentence in sentences if sentence in summary_sentences]

    return " ".join(summary)


# ---------------- Home Page ----------------

@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""
    input_text = ""

    if request.method == "POST":

        input_text = request.form.get("text", "")

        if input_text.strip():

            summary = frequency_summary(input_text, 3)

    return render_template(
        "index.html",
        summary=summary,
        input_text=input_text
    )


# ---------------- Download Summary ----------------

@app.route("/download", methods=["POST"])
def download():

    summary = request.form.get("summary", "")

    file_path = "summary.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(summary)

    return send_file(
        file_path,
        as_attachment=True,
        download_name="Generated_Summary.txt"
    )


if __name__ == "__main__":
    app.run(debug=True)
