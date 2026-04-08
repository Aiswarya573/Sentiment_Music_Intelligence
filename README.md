#  Sentiment_Music_Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-Naïve%20Bayes-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-Tkinter-green?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Detect your mood. Discover your music. 🎧**

*An ML-powered system that analyzes your emotional state from text and recommends music that matches your vibe.*

</div>
## ⚡ Key Highlights

- Built using **Machine Learning + NLP**
- Real-time **sentiment detection with confidence**
- Smart **music recommendation system**
- Beginner-friendly **GUI application**

---

##  Overview

The **Sentiment-Driven Music Intelligence System** is a Python-based Machine Learning mini-project that:

- Takes natural language text as input (e.g., *"I'm feeling really happy today"*)
- Classifies the sentiment using a **Naïve Bayes + Lexicon Ensemble Model**
- Recommends matching **music genres and songs**
- Displays results in a sleek **Tkinter dark-theme GUI**

---

##  Features

| Feature | Description |
|---|---|
|  **ML Sentiment Model** | Multinomial Naïve Bayes trained on a seed corpus |
|  **Ensemble Analysis** | Combines Naïve Bayes + Lexicon for higher accuracy |
|  **Negation Handling** | Understands *"not happy"* → Negative |
|  **Intensifiers** | Boosts score for *"very"*, *"extremely"*, *"so"* etc. |
|  **Dark Theme GUI** | Modern Tkinter interface with live probability bars |
|  **Confidence Score** | Shows prediction confidence % for each class |
|  **Music Recommendations** | Genre + Song suggestions based on detected mood |

---

##  Algorithm

### Ensemble Approach (NB + Lexicon)

```
User Input
    │
    ├──► Naïve Bayes Model ──► NB Label + Confidence
    │         (trained on seed corpus)
    │
    ├──► Lexicon Scorer ──────► Lexicon Label + Score
    │         (negation + intensifier aware)
    │
    └──► Ensemble Logic ──────► Final Label + Confidence %
              │
              └──► Music Recommender ──► Genres + Songs
```

**Ensemble Rule:**
- If **both agree** → final label = that label, confidence boosted
- If **they disagree** → NB wins when confidence > 55%, else Lexicon wins

---

##  GUI Preview

```
┌─────────────────────────────────────────────────────┐
│  Mood Music                                       │
│ Tell me how you feel — I'll find music for you      │
├─────────────────────────────────────────────────────┤
│ How are you feeling today?                          │
│ [ I'm feeling really happy today!        ]          │
│ [ Analyze Mood → ]                                  │
├─────────────────────────────────────────────────────┤
│    POSITIVE                                       │
│      Confidence: 94.2% | NB: positive | Lex: pos   │
│                                                     │
│ Positive ████████████████████░░░░░░░  78.4%        │
│ Negative ░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.1%        │
│ Neutral  ██░░░░░░░░░░░░░░░░░░░░░░░░░  13.5%        │
├─────────────────────────────────────────────────────┤
│  Recommended Music                                │
│ Genres          │ Songs                             │
│ ● Pop           │ ♪ Shape of You - Ed Sheeran       │
│ ● Dance         │ ♪ Blinding Lights - The Weeknd    │
│ ● EDM           │ ♪ Uptown Funk - Bruno Mars        │
└─────────────────────────────────────────────────────┘
```
##  Real Application Screenshot

![App Screenshot](screenshot.png)

---

##  Project Structure

```
sentiment-driven-music-intelligence-system/
│
├──  dataset/
│   └── sentiment140.csv          # Sentiment140 dataset (download separately)
│
├──  src/
│   ├── preprocess.py             # Text cleaning + dataset loader
│   ├── sentiment_model.py        # Naïve Bayes + Lexicon ensemble
│   └── music_recommender.py      # Mood → Genre/Song mapper
│
├── main.py                       # CLI entry point
├── gui.py                        # Tkinter GUI entry point  ← NEW
├── requirements.txt
└── README.md
```

---

##  Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aiswarya573/Sentiment_Music_Intelligence.git
cd Sentiment_Music_Intelligence
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3A: Run with GUI *(Recommended)*

```bash
python gui.py
```

### Step 3B: Run with CLI

```bash
python main.py
```

---

##  Dataset

This project uses the **[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)** (1.6 million tweets).

>  The dataset is **not included** in the repo due to size. Download it from Kaggle.

After downloading, place it here:
```
dataset/sentiment140.csv
```

---

##  Example Output

**Input:** *"I'm feeling really happy and excited today!"*

```
Detected Sentiment : POSITIVE
Confidence         : 94.2%
NB Prediction      : positive
Lexicon Score      : +3.0

Recommended Genres : Pop, Dance, EDM
Recommended Songs  :
  ♪ Shape of You - Ed Sheeran
  ♪ Blinding Lights - The Weeknd
  ♪ Uptown Funk - Bruno Mars
```

---

##  Sentiment Classes

| Class | Trigger Words (examples) | Music Style |
|---|---|---|
|  **Positive** | happy, love, awesome, great, excited | Pop, Dance, EDM |
|  **Neutral** | okay, fine, normal, usual | Jazz, Classical |
|  **Negative** | sad, angry, hate, terrible, stressed | Lo-fi, Acoustic |

---

##  Technologies Used

- **Python 3.8+**
- **Pandas** — Dataset handling
- **NumPy** — Numerical computations
- **Tkinter** — GUI (built-in with Python)
- **Math / Collections** — Naïve Bayes implementation (no sklearn dependency)

---
##  Future Improvements

- [ ] Integrate **Spotify API** for real-time song recommendations
- [ ] Train on the full **Sentiment140 dataset** for better accuracy
- [ ] Add **emoji sentiment detection** 
- [ ] Support **multilingual** sentiment analysis
- [ ] Build a **web app** version with Flask/Streamlit
- [ ] Add **voice input** support

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

##  License

This project is licensed under the **MIT License**.

---

<div align="center">

Made with  | Machine Learning Mini Project

</div>
##  Author

**Aiswarya Tech**  
  GitHub   : https://github.com/Aiswarya573
  LinkedIn :https://www.linkedin.com/in/aiswarya-m-257002381/
  
