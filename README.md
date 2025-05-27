# Speech-to-text-converter
# 🎧 Audio Analysis Mini Project

### 📁 Project Title: Audio Transcription, Sentiment Analysis & Topic Modeling  

---

## 📌 Project Overview

This project presents a system that takes an **audio file**, converts it into **text**, detects **emotions** expressed in the content, and extracts the **main topics** discussed. The audio used for this case study was titled **"Organs of the Human Body"**, which contains information about various organs, their locations, and functions.

---

## 🧪 Methodology

The system was developed using **Python** with a **Flask** backend. The workflow includes:

1. **Audio Transcription**  
   - Audio is converted to text using a speech recognition library.
   - Files are converted to `.wav` format for compatibility.
   
2. **Sentiment Analysis**  
   - Performed using a pre-trained model from **Hugging Face**.
   - Output: **NEGATIVE** sentiment with 0.98 confidence, likely skewed by words like *waste* and *deoxygenated*.

3. **Topic Modeling**  
   - Implemented using **LDA (Latent Dirichlet Allocation)** from `scikit-learn`.
   - Topics were extracted by identifying frequently occurring words.

---

## 🧾 Results

### 💬 Sentiment Analysis:
- Classified as: **NEGATIVE**  
- Note: Although the audio content was educational, certain words triggered negative classification.

### 📚 Topics Identified:
- **Topic 1:** `body`, `blood`, `intestine` – Related to bodily processes, circulation, and digestion.
- **Topic 2:** `exhaling`, `example`, `work` – Related to respiratory system and functional examples.

---

## ✅ Conclusion

This mini project demonstrates a simple but effective pipeline for:
- Converting speech to text
- Performing emotion detection on content
- Extracting main discussion themes

While sentiment analysis may be misled by scientific vocabulary, topic modeling provided meaningful insights. The system has promising applications in **educational content analysis**, **lecture indexing**, and **audio summarization**.

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **SpeechRecognition** (for audio-to-text)
- **Transformers** (Hugging Face models)
- **Scikit-learn** (LDA for topic modeling)

---

## 📁 How to Run

1. Upload an audio file (.wav)
2. The system processes it and outputs:
   - Transcribed Text
   - Sentiment Score
   - Extracted Topics

> 💡 This project can be extended to multi-language support, improved sentiment accuracy, and visual topic clustering.

---

