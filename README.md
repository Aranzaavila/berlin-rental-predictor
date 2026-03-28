# 🏠 Berlin Rental Price Predictor

Berlin Rental Price Predictor is a machine learning web application that estimates Airbnb nightly prices in Berlin based on key listing details such as neighbourhood, room type, number of guests, and bedrooms. The app uses a trained Random Forest model and presents predictions through an interactive Streamlit interface, making it easy to explore how different property features influence expected rental prices.

## Live Demo

[View the live demo](https://berlin-rental-predictor.streamlit.app)

## Features

- Real Airbnb data from 14,000+ Berlin listings
- Random Forest ML model
- Interactive Streamlit web interface
- Predicts nightly price based on neighbourhood, room type, guests and bedrooms

## Tech Stack

- Python
- scikit-learn
- pandas
- Streamlit
- joblib

## How to Run Locally

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate the trained model:

```bash
python src/train.py
```

5. Start the Streamlit app:

```bash
python -m streamlit run app.py
```

## Model Performance

The model achieved an R² score of 0.54 when trained on Berlin Airbnb data.

## Dataset

This project uses Airbnb listing data for Berlin provided by [Inside Airbnb](https://insideairbnb.com).
