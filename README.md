# Enhanced-Social-Media-Sentiment-Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HDHqin2Mm2vP2iDbb7ztQylTse7e0b_L?usp=sharing)

## Overview

This project is a social media sentiment analysis application built using machine learning, Natural Language Processing (NLP), and a Gradio interface. The solution allows users to input text and receive sentiment predictions, confidence scores, effectiveness ratings, improvement suggestions, and a graphical representation of sentiment confidence. The model is trained on social media data, and the results can be utilized for improving engagement, content clarity, and sentiment-driven marketing strategies.

### Key Features:
- Sentiment Prediction: Determines if the text is positive, neutral, or negative.
- Confidence Score (%): Provides a percentage confidence level in the prediction.
- Effectiveness Score (%): Measures the effectiveness of the text, influenced by sentiment and clarity.
- Improvement Suggestions: Offers actionable insights to enhance the sentiment or clarity of the text.
- Sentiment Confidence Plot: Displays a bar graph showing the confidence levels for each sentiment class.

### Project Category:
This solution falls under the **Open Innovation** track as it involves AI-powered sentiment analysis applied to social media text, which can be used across various domains such as marketing, customer feedback analysis, and content improvement.

---

## Requirements

### Prerequisites
To run this project locally, ensure you have the following dependencies installed:

1. **Python**: Version 3.6 or higher.
2. **Libraries**: You can install the required libraries using `pip`.

```bash
pip install joblib gradio numpy scikit-learn textblob plotly
```

- **`joblib`**: For loading and saving machine learning models.
- **`gradio`**: For creating the user interface and deploying the sentiment analysis tool.
- **`numpy`**: For numerical operations and array manipulation.
- **`scikit-learn`**: For machine learning models and text vectorization.
- **`textblob`**: For sentiment analysis and effectiveness scoring.
- **`plotly`**: For generating interactive confidence plots.

---

## How It Works

1. **Text Preprocessing**: The input text is preprocessed by converting it to lowercase and removing punctuation using a regular expression.
2. **Sentiment Prediction**: The preprocessed text is vectorized using the TF-IDF vectorizer and fed into the trained machine learning model to predict the sentiment (positive, neutral, or negative).
3. **Confidence Score**: The model outputs a probability for each sentiment class. The highest probability is used to calculate the confidence score (in percentage).
4. **Effectiveness Score**: The effectiveness of the text is estimated based on the sentiment score and the clarity of the text using TextBlob, which analyzes the polarity of the text.
5. **Improvement Suggestions**: Based on the effectiveness score and sentiment, suggestions are made to improve the text for better clarity or engagement.
6. **Confidence Plot**: A Plotly bar graph is generated to visually display the sentiment class confidence levels.

---

## Automatic Setup Instructions

1. To Get Started Click the Open in Colab At the start of the README
2. Conect to the Colab Servers
3. Run Each Cell One by One
4. After the last cell click on Gradio link

Once the application runs, it will open a local Gradio interface in your browser where you can input text and interact with the sentiment analysis tool.

---

## Manual Setup Instructions

Follow these steps to set up the sentiment analysis project on your local machine or cloud environment.

### Step 1: Install Required Libraries

First, ensure that you have Python 3.7+ installed. You'll need the following Python libraries:

```bash
pip install joblib
pip install gradio
pip install plotly
pip install numpy
pip install textblob
```

### Step 2: Download the Pretrained Model and Vectorizer

You can download the pretrained model and vectorizer directly from the provided links:

```bash
!wget https://github.com/AshbilShahid/Enhanced-Social-Media-Sentiment-Analysis/raw/main/sentiment_model.pkl
!wget https://github.com/AshbilShahid/Enhanced-Social-Media-Sentiment-Analysis/raw/main/vectorizer.pkl
```

### Step 3: Load the Model and Vectorizer

Load the pretrained model and vectorizer by running the following code in your notebook:

```python
import joblib

# Load the trained model and vectorizer
try:
    model = joblib.load('/content/sentiment_model.pkl')
    vectorizer = joblib.load('/content/vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
```

### Step 4: Define the Sentiment Prediction Function

Now, define the function that will preprocess text, generate sentiment predictions, and calculate the confidence and effectiveness scores:

```python
import numpy as np
import re
from textblob import TextBlob
import plotly.graph_objects as go

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_sentiment(text):
    try:
        # Preprocess the input text
        clean_text = preprocess_text(text)
        
        # Vectorize and predict
        text_tfidf = vectorizer.transform([clean_text])
        prediction = model.predict(text_tfidf)[0]
        prediction_proba = model.predict_proba(text_tfidf)[0]
        confidence_score = round(np.max(prediction_proba) * 100, 2)  # Confidence in %

        # Effectiveness Calculation using TextBlob
        polarity = TextBlob(clean_text).sentiment.polarity
        effectiveness_score = max(0, min(100, confidence_score + (polarity * 20)))

        # Improvement Suggestions
        if effectiveness_score < 60:
            improvement_suggestions = "Consider clearer, more vivid language to convey sentiment."
        elif prediction == "positive":
            improvement_suggestions = "Add emotional depth for better engagement."
        elif prediction == "negative":
            improvement_suggestions = "Add specific examples or context to support your point."
        else:
            improvement_suggestions = "Try focusing on clarity for a stronger message."

        # Plotly Confidence Plot
        labels = ["Negative", "Neutral", "Positive"]
        fig = go.Figure(data=[go.Bar(x=labels, y=prediction_proba, marker_color=['red', 'gray', 'green'])])
        fig.update_layout(
            title="Sentiment Confidence Scores",
            xaxis_title="Sentiment",
            yaxis_title="Confidence",
            yaxis=dict(range=[0, 1])
        )

        return prediction, f"{confidence_score}%", f"{effectiveness_score}%", improvement_suggestions, fig
    
    except Exception as e:
        print(f"Error in prediction function: {e}")
        return "Error", "Error", "Error", "Error", go.Figure()
```

### Step 5: Set Up the Gradio Interface

Create a Gradio interface to allow users to interact with the model:

```python
import gradio as gr

# Set up the Gradio interface with proper outputs
interface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs=[
        gr.Textbox(label="Sentiment Prediction"),
        gr.Textbox(label="Confidence Score (%)"),
        gr.Textbox(label="Effectiveness Score (%)"),
        gr.Textbox(label="Improvement Suggestions"),
        gr.Plot(label="Confidence Plot")
    ],
    title="Enhanced Social Media Sentiment Analysis",
    description="Enter text to analyze its sentiment, confidence, effectiveness, and get suggestions for improvement.",
    examples=["I love this product!", "This is the worst experience I've ever had.", "I'm not sure about this..."]
)

# Launch the Gradio interface
interface.launch()
```

### Step 6: Run the Application

Once everything is set up, run the application by executing the notebook. Gradio will launch an interactive interface in your browser where you can input text, and the model will return sentiment predictions along with additional metrics and suggestions.


---
## Google Technologies Contributed

- **TensorFlow**: While not directly used in the provided code, TensorFlow is a common framework for machine learning and have been used in the original model development.
- **Google Colab**: Provided an easy-to-use platform for testing and development with cloud resources. TensorFlow, Plotly, and other libraries can be used seamlessly within Colab.
- **Google Cloud Storage**: Used for storing the sentiment analysis model and vectorizer, ensuring that they can be easily accessed in the Colab environment.
- 

---

## Future Enhancements

- **Integration with Google Cloud Storage**: For storing large datasets or model files.
- **Scaling with Google Cloud Functions**: Enable scaling of the model inference as a serverless solution.
- **Advanced Sentiment Models**: Integration of more advanced deep learning models, such as BERT or GPT, to enhance sentiment analysis accuracy.
- **Cloud Database**: Store historical sentiment predictions using Google Firebase or Cloud Firestore to track sentiment trends over time.

---

## Troubleshooting

If you encounter any errors, check the following:

1. **Model Loading Errors**: Ensure the paths to the `sentiment_model.pkl` and `vectorizer.pkl` are correct and the files are not corrupted.
2. **Gradio Interface Issues**: Make sure that the required libraries (`gradio`, `numpy`, etc.) are installed correctly.
3. **Plot Rendering Issues**: If the Plotly graph is not rendering correctly, ensure that your browser supports interactive JavaScript plots.

For additional support, consult the official documentation for **Gradio**, **scikit-learn**, and **Plotly**.

---

## License

This project is licensed under the ASH License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Gradio**: For creating a simple and powerful interface for machine learning models.
- **scikit-learn**: For providing easy-to-use tools for machine learning and NLP.
- **TensorFlow**: For its potential use in deep learning models and expanding AI capabilities.
- **Plotly**: For interactive visualizations, making it easier to understand model outputs.

---
