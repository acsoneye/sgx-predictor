import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SGX Stock Up/Down Screener", layout="wide")
st.title("🇸🇬 SGX Market Screener: Next-Day Up/Down Prediction")

STI_STOCKS = {
    "DBS Group (D05.SI)": "D05.SI",
    "OCBC Bank (O39.SI)": "O39.SI",
    "UOB Bank (U11.SI)": "U11.SI",
    "Singtel (Z74.SI)": "Z74.SI",
    "Keppel Ltd (BN4.SI)": "BN4.SI",
    "ST Engineering (S63.SI)": "S63.SI",
    "SGX Ltd (S68.SI)": "S68.SI",
    "Singapore Airlines (C6L.SI)": "C6L.SI",
    "CapitaLand IntCom Trust (C38U.SI)": "C38U.SI",
    "CapitaLand Investment (9CI.SI)": "9CI.SI",
    "Yangzijiang Shipbuilding (BS6.SI)": "BS6.SI",
    "Sembcorp Industries (U96.SI)": "U96.SI",
    "SATS Ltd (S58.SI)": "S58.SI",
    "City Developments (C09.SI)": "C09.SI",
    "Genting Singapore (G13.SI)": "G13.SI",
}

st.sidebar.header("Scan Parameters")
history_period = st.sidebar.selectbox("Training Data Period", ["1y", "2y"], index=0)
confidence_threshold = st.sidebar.slider("Minimum Confidence Filter (%)", 50, 75, 55)

def calculate_indicators(df):
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['Return_1d'] = data['Close'].pct_change()
    data['Return_5d'] = data['Close'].pct_change(5)
    data['Volatility_10d'] = data['Return_1d'].rolling(10).std()

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data

def train_and_predict(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty or len(df) < 60:
        return None

    clean_df = calculate_indicators(df)
    features = ['SMA_5', 'SMA_20', 'Return_1d', 'Return_5d', 'Volatility_10d', 'RSI', 'Volume']
    valid = clean_df.dropna().copy()
    if len(valid) < 40:
        return None

    X = valid[features].iloc[:-1]
    y = valid['Target'].iloc[:-1]
    X_latest = valid[features].iloc[-1:]
    last_close = float(valid['Close'].iloc[-1])

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y)

    prob = clf.predict_proba(X_latest)[0]
    prob_down, prob_up = prob[0], prob[1]

    signal = "🟢 UP" if prob_up >= 0.5 else "🔴 DOWN"
    confidence = max(prob_up, prob_down) * 100

    return {
        "Ticker": ticker,
        "Last Close (SGD)": round(last_close, 2),
        "Predicted Direction": signal,
        "Confidence (%)": round(confidence, 1),
        "P(UP) %": round(prob_up * 100, 1),
        "P(DOWN) %": round(prob_down * 100, 1),
        "RSI": round(float(valid['RSI'].iloc[-1]), 1)
    }

if st.button("🚀 Run SGX Prediction Scan", use_container_width=True):
    with st.spinner("Fetching SGX live data and calculating predictions..."):
        results = []
        for name, ticker in STI_STOCKS.items():
            res = train_and_predict(ticker, history_period)
            if res:
                res["Stock Name"] = name
                results.append(res)

        if results:
            res_df = pd.DataFrame(results)
            cols = ["Stock Name", "Ticker", "Last Close (SGD)", "Predicted Direction", "Confidence (%)", "P(UP) %", "P(DOWN) %", "RSI"]
            res_df = res_df[cols]

            up_count = len(res_df[res_df["Predicted Direction"] == "🟢 UP"])
            down_count = len(res_df[res_df["Predicted Direction"] == "🔴 DOWN"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Stocks Scanned", len(res_df))
            col2.metric("Predicted UP 🟢", up_count)
            col3.metric("Predicted DOWN 🔴", down_count)

            filtered_df = res_df[res_df["Confidence (%)"] >= confidence_threshold]

            st.subheader(f"Predictions (Confidence ≥ {confidence_threshold}%)")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                res_df,
                x="Ticker",
                y="P(UP) %",
                color="Predicted Direction",
                color_discrete_map={"🟢 UP": "#2ca02c", "🔴 DOWN": "#d62728"},
                title="Predicted Chance of Price Going UP Tomorrow (%)",
                labels={"P(UP) %": "Chance of UP (%)"}
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Line")
            fig.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click the button above to run the scan across Singapore blue chips.")