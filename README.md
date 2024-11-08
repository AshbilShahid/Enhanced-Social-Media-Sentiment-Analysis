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



---

Once the application runs, it will open a local Gradio interface in your browser where you can input text and interact with the sentiment analysis tool.

---

## How It Works

1. **Text Preprocessing**: The input text is preprocessed by converting it to lowercase and removing punctuation using a regular expression.
2. **Sentiment Prediction**: The preprocessed text is vectorized using the TF-IDF vectorizer and fed into the trained machine learning model to predict the sentiment (positive, neutral, or negative).
3. **Confidence Score**: The model outputs a probability for each sentiment class. The highest probability is used to calculate the confidence score (in percentage).
4. **Effectiveness Score**: The effectiveness of the text is estimated based on the sentiment score and the clarity of the text using TextBlob, which analyzes the polarity of the text.
5. **Improvement Suggestions**: Based on the effectiveness score and sentiment, suggestions are made to improve the text for better clarity or engagement.
6. **Confidence Plot**: A Plotly bar graph is generated to visually display the sentiment class confidence levels.

---
---

## Future Enhancements

- **Integration with Google Cloud Storage**: For storing large datasets or model files.
- **Scaling with Google Cloud Functions**: Enable scaling of the model inference as a serverless solution.
- **Advanced Sentiment Models**: Integration of more advanced deep learning models, such as BERT or GPT, to enhance sentiment analysis accuracy.
- **Cloud Database**: Store historical sentiment predictions using Google Firebase or Cloud Firestore to track sentiment trends over time.

---

## Troubleshooting

If you encounter any errors, check the following:

1. **Model Loading Errors**: Ensure the paths to the `1sentiment_model.pkl` and `tfidf_vectorizer.pkl` are correct and the files are not corrupted.
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
