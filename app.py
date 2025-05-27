from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

import os
from pydub import AudioSegment
import logging

# Setup
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    sentiment = None 
    return render_template('index.html', sentiment=sentiment)  # Serving the frontend HTML

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided in the request.'}), 400
        
        # Get the uploaded file
        audio_file = request.files['audio_file']
        
        # Save the file temporarily
        temp_audio_path = os.path.join("uploads", audio_file.filename)
        audio_file.save(temp_audio_path)
        
        # Convert the file to WAV if it's not already in WAV format
        if not temp_audio_path.endswith(".wav"):
            audio = AudioSegment.from_file(temp_audio_path)
            temp_audio_path = temp_audio_path.replace(".mp3", ".wav")
            audio.export(temp_audio_path, format="wav")
        
        # Now, process the audio (transcription and sentiment analysis)
        text = transcribe_audio(temp_audio_path)
        sentiment = analyze_sentiment(text)
        topics = extract_topics(text)
        
        
        return jsonify({
            'status': 'success',
            'message': 'Audio processed successfully!',
            'transcribed_text': text,
            'sentiment': sentiment,
            'topics': topics
        })
    
    except Exception as e:
        app.logger.error(f"Error processing audio: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# Audio transcription function
def transcribe_audio(audio_path):
    # Use SpeechRecognition to transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        # Using Google's speech-to-text API (you can replace this with any transcription service)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Audio not understood."
    except sr.RequestError as e:
        return f"Error with transcription service: {e}"


# Sentiment analysis (using HuggingFace's transformers)
from transformers import pipeline

# Sentiment analysis (using HuggingFace's transformers)
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    
    # Extract the label (positive/negative) and the score (confidence)
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']
    
    return {
        'label': sentiment_label,  # Positive/Negative
        'score': round(sentiment_score, 4)  # Rounded to 4 decimal places
    }


def extract_topics(text, num_topics=3, num_words=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_vector = vectorizer.fit_transform([text])
    lda = LDA(n_components=num_topics, random_state=0)
    lda.fit(text_vector)
    
    words = vectorizer.get_feature_names_out()
    topic_words = [
        [words[i] for i in topic.argsort()[-num_words:]] for topic in lda.components_
    ]
    
    # Return topics 
    topic_freq = {f'Topic {i+1}': ' '.join(topic_words[i]) for i in range(len(topic_words))}
    return topic_freq


if __name__ == "__main__":
    app.run(debug=True, port=5000)
