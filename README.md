# Enhanced-Social-Media-Sentiment-Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HDHqin2Mm2vP2iDbb7ztQylTse7e0b_L?usp=sharing)
## Overview
This project is a social media sentiment analysis application built using machine learning, Natural Language Processing (NLP), and a Gradio interface. The solution allows users to input text and receive sentiment predictions, confidence scores, effectiveness ratings, improvement suggestions, and a graphical representation of sentiment confidence. The model is trained on social media data, and the results can be utilized for improving engagement, content clarity, and sentiment-driven marketing strategies.

## Key Features:
- Sentiment Prediction: Determines if the text is positive, neutral, or negative.
- Confidence Score (%): Provides a percentage confidence level in the prediction.
- Effectiveness Score (%): Measures the effectiveness of the text, influenced by sentiment and clarity.
- Improvement Suggestions: Offers actionable insights to enhance the sentiment or clarity of the text.
- Sentiment Confidence Plot: Displays a bar graph showing the confidence levels for each sentiment class.
## Project Category:
This solution falls under the Open Innovation track as it involves AI-powered sentiment analysis applied to social media text, which can be used across various domains such as marketing, customer feedback analysis, and content improvement.
## Requirements
### Prerequisites
To run this project locally, ensure you have the following dependencies installed:

i. Python: Version 3.6 or higher.
2. Libraries: You can install the required libraries using pip.
bash
code(pip install joblib gradio numpy scikit-learn textblob plotly)
joblib: For loading and saving machine learning models.
gradio: For creating the user interface and deploying the sentiment analysis tool.
numpy: For numerical operations and array manipulation.
scikit-learn: For machine learning models and text vectorization.
textblob: For sentiment analysis and effectiveness scoring.
plotly: For generating interactive confidence plots.

