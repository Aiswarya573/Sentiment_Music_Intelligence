# Sentiment-Driven Music Intelligence System

## Project Overview

The **Sentiment-Driven Music Intelligence System** is a Machine Learning based mini-project that analyzes a user's mood from text input and recommends suitable music genres and songs.

The system performs **sentiment analysis** on the user's input and classifies the emotion into:

* Positive
* Neutral
* Negative

Based on the detected sentiment, the system suggests appropriate **music genres and songs**.

---

##  Features

* Detects user mood from text input
* Classifies sentiment (Positive / Neutral / Negative)
* Recommends music genres
* Suggests songs based on mood
* Simple command-line interface

---

##  Algorithm Used

This project currently uses **Rule-Based Sentiment Analysis**.

The algorithm detects sentiment by analyzing keywords in the user input and mapping them to predefined sentiment categories.

Workflow:

User Input → Sentiment Detection → Mood Classification → Music Recommendation

---

##  Technologies Used

* Python
* Pandas
* NumPy
* SciPy

Python libraries used:

* pandas
* numpy
* scipy

---

##  Project Structure

```
sentiment-driven-music-intelligence-system
│
├── dataset
│
├── src
│   ├── preprocess.py
│   ├── sentiment_model.py
│   ├── music_recommender.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

##  Dataset

This project uses the **Sentiment140 Dataset**.

Download it from Kaggle:

https://www.kaggle.com/datasets/kazanova/sentiment140

After downloading, place the dataset file inside the dataset folder:

```
dataset/sentiment140.csv
```

---

##  How to Run the Project

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/sentiment-driven-music-intelligence-system.git
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the Program

```
python main.py
```

---

##  Example Output

```
Sentiment Driven Music Intelligence System

How are you feeling today? happy

Detected Sentiment: positive

Recommended Genre:
- Pop
- Dance
- EDM

Recommended Songs:
- Shape of You - Ed Sheeran
- Blinding Lights - The Weeknd
- Uptown Funk - Bruno Mars
```

---

## Future Improvements

* Use Machine Learning algorithms like Naive Bayes
* Integrate Spotify API for real music recommendations
* Build a graphical user interface
* Improve sentiment detection accuracy

---

## 👨‍💻Author

Machine Learning Mini Project
