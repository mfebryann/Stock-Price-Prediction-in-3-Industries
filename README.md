# 📈 Stock Price Prediction with LSTM

A Deep Learning project that implements stock price prediction for Indonesian stocks (IDX) using **Long Short-Term Memory (LSTM)** neural networks. The system automatically fetches historical price data, preprocesses it, builds and trains an LSTM model, and forecasts future stock prices while excluding weekends.

---

## 🌟 Key Features

* **Data Acquisition:** Automatically downloads historical stock data using `yfinance`.
* **Data Preprocessing:** Extracts closing prices, scales data using `MinMaxScaler`, and formats it into 30-day sliding window sequences.
* **Model Architecture:** Multi-layer sequential LSTM network with Dropout layers to prevent overfitting.
* **Weekend-Aware Forecasting:** Predicts future trading days while skipping weekends.
* **Performance Evaluation:** Evaluates model accuracy using Root Mean Squared Error (RMSE).
* **Visualization:** Generates clear line charts comparing predicted vs. actual prices.
* **Multi-Stock Support:** Tested on major Indonesian stocks such as **BBCA**, **BBRI**, **ISAT**, **MIKA**, **TLKM**, and **PYFA**.

---

## 🏗️ Project Pipeline

1. **Data Download:** Fetches daily stock data via Yahoo Finance (`yfinance`).
2. **Preprocessing & Feature Engineering:**
   * Extracts `"Close"` prices.
   * Scales feature values between 0 and 1 using `MinMaxScaler`.
   * Formats data into input sequences using a **30-day window**.
3. **Train / Test Split:** Allocates **80%** of data for training and **20%** for testing.
4. **LSTM Model Construction:**
   * 3 to 4 LSTM layers paired with Dropout layers for regularization.
   * Dense output layer predicting the next closing price.
   * Compiled using the `Adam` optimizer and `Mean Squared Error` (MSE) loss.
5. **Training Setup:**
   * Epochs: `200` | Batch size: `16`
   * Reproducibility set via fixed random seeds (`np.random.seed(42)` & `tf.random.set_seed(42)`).

---

## 🚀 Getting Started

### Prerequisites & Installation

Ensure you have Python 3.x installed, then install the required dependencies:

`pip install yfinance pandas numpy tensorflow keras matplotlib scikit-learn`

### How to Run

1. Configure your target stock ticker and date range inside your Python script:

`ticker = "BBCA.JK"`
`start_date = "2021-12-31"`
`end_date = "2023-12-18"`

2. Execute the script:
   * The pipeline will train the LSTM model, evaluate it using RMSE, predict the next 22 trading days (skipping weekends), and save outputs/plots automatically.

---

## 📂 Outputs & Results

Running the model produces the following artifacts:

* **CSV Data Exports:**
  * Training dataset split (e.g., `BCA_train.csv`)
  * Testing dataset split with predictions (e.g., `BCA_testing.csv`)
  * Final merged output (e.g., `BCA.csv`)
* **Visualizations:** Line charts comparing actual vs. predicted prices over time.
* **Future Predictions:** A formatted DataFrame containing multi-step forecasts:

| Tanggal | Prediksi_Harga |
| :--- | :--- |
| **2023-12-18** | 9139.93 |
| **2023-12-19** | 9108.79 |
| **...** | ... |

---

## 📊 Model Architecture & Performance

### Code Highlight: LSTM Architecture

`model = keras.Sequential()`
`model.add(layers.LSTM(50, activation='tanh', return_sequences=True, input_shape=(30, 1)))`
`model.add(layers.Dropout(0.2))`
`model.add(layers.LSTM(50, return_sequences=True))`
`model.add(layers.Dropout(0.2))`
`model.add(layers.LSTM(60, return_sequences=True))`
`model.add(layers.Dropout(0.2))`
`model.add(layers.LSTM(30, return_sequences=False))`
`model.add(layers.Dropout(0.2))`
`model.add(layers.Dense(1))`

`model.compile(optimizer='adam', loss='mean_squared_error')`
`model.fit(x_train, y_train, batch_size=16, epochs=200)`

### Evaluation Metrics

Sample Root Mean Squared Error (RMSE) ranges across tested stocks:

* **Train RMSE:** ~30 to ~38
* **Test RMSE:** ~50 to ~100 *(varies based on individual stock volatility)*

---

## 🛠️ Built With

* [Python 3.x](https://www.python.org/) - Programming Language
* [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance Data Downloader
* [TensorFlow / Keras](https://www.tensorflow.org/) - Deep Learning Framework
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Manipulation
* [Scikit-learn](https://scikit-learn.org/) - Data Scaling & Evaluation Metrics
* [Matplotlib](https://matplotlib.org/) - Data Visualization

---

## 📌 Notes & Future Improvements

* The model relies solely on historical daily closing prices (`Close`) and does not incorporate technical indicators or external market sentiment.
* Network depth, sequence length (window size), and hyperparameters can be fine-tuned per stock for improved accuracy.
