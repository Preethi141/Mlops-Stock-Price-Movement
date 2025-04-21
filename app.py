from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and training column order
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    ticker = request.form['ticker']
    sentiment = float(request.form['sentiment'])
    volume = int(request.form['volume'])
    open_price = float(request.form['open_price'])
    close_price = float(request.form['close_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])

    # Construct input data
    input_data = {
        "sentiment_score": sentiment,
        "volume": volume,
        "open_price": open_price,
        "close_price": close_price,
        "high_price": high_price,
        "low_price": low_price,
        "ticker_AAPL": 0,
        "ticker_AMZN": 0,
        "ticker_GOOG": 0,
        "ticker_GOOG": 0,
        "ticker_MSFT": 0,
        "ticker_TSLA": 0,
        f"ticker_{ticker}": 1  # Activate the chosen one
    }

    # Fix any missing columns
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    # Align with training column order
    input_df = pd.DataFrame([input_data])[model_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template("predict.html", date=date, ticker=ticker, sentiment=sentiment, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
