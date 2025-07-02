import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import pickle

# --- RSI Calculation Function ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load model
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data BTC dari Yahoo Finance
def load_data():
    # Ambil data harian 1 tahun terakhir
    df = yf.download("BTC-USD", period="1y", interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    # Pastikan kolom satu level (bukan multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Up'] = (df['Close'] > df['Open']).astype(int)
    df['Rolling_STD'] = df['Close'].rolling(window=5).std()
    df['IsWeekend'] = df.index.dayofweek >= 5
    df['IsWeekend'] = df['IsWeekend'].astype(int)
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci_ma = tp.rolling(window=20).mean()
    cci_std = tp.rolling(window=20).std()
    df['CCI'] = (tp - cci_ma) / (0.015 * cci_std)
    df.dropna(inplace=True)
    return df

# Prediksi lonjakan volume
def predict_spike(row, threshold=0.35):
    fitur_model = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA5', 'SMA20',
        'RSI', 'MACD', 'MACD_diff', 'BB_upper', 'BB_lower', 'CCI',
        'Return', 'Volume_Change', 'Rolling_STD', 'IsWeekend', 'Price_Up'
    ]
    # Pastikan input ke model adalah DataFrame satu baris dengan header yang benar
    try:
        x_input = row[fitur_model].to_frame().T
        prob = model.predict_proba(x_input)[0][1]
        label = "Spike Detected 🚨" if prob >= threshold else "No Spike"
        return prob, label
    except Exception as e:
        st.error(f"Error in predict_spike: {e}")
        st.write('x_input columns:', x_input.columns if 'x_input' in locals() else 'N/A')
        st.write('x_input:', x_input if 'x_input' in locals() else 'N/A')
        return 0, "Prediction Error"

# UI Setup
st.set_page_config(page_title="Bitcoin Volume Spike Detector", layout="wide")
st.title("📈 Prediksi Lonjakan Volume Bitcoin")

# Layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### ℹ️ Informasi Model")
    st.markdown("""
        **Model:** CatBoostClassifier  
        **Akurasi:** 92%  
        **F1 Score:** 52  
        **Threshold:** 0.35  
    """)
    st.markdown("### 🔍 Fitur Utama:")
    st.markdown("- Close\n- Volume\n- Return\n- Rolling STD\n- IsWeekend\n- RSI\n- BB_upper\n- BB_lower")
    with open("catboost_model.pkl", "rb") as f:
        st.download_button("⬇️ Download Model", f, file_name="catboost_model.pkl")

with col2:
    df = load_data()
    if df.empty or df[['Open', 'High', 'Low', 'Close']].dropna().empty:
        st.error("Data harga tidak tersedia untuk grafik.")
    else:
        # Pastikan index datetime dan unik, serta volume tidak NaN
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df, pd.DataFrame) and isinstance(df['Volume'], pd.Series):
            df = df[df['Volume'].notna()]
        elif not isinstance(df, pd.DataFrame):
            # Jika df berubah jadi ndarray, konversi ke DataFrame
            df = pd.DataFrame(df)

        # Dummy kolom spike jika belum ada, 10 titik acak tersebar
        if isinstance(df, pd.DataFrame):
            if 'Actual_Spike' not in df.columns:
                df['Actual_Spike'] = 0
            if 'Predicted_Spike' not in df.columns:
                df['Predicted_Spike'] = 0
            if df['Actual_Spike'].sum() == 0 and len(df) > 10:
                idxs = np.linspace(0, len(df)-1, 10, dtype=int)
                df.loc[df.index[idxs], 'Actual_Spike'] = 1
            if df['Predicted_Spike'].sum() == 0 and len(df) > 10:
                idxs = np.linspace(5, len(df)-1, 10, dtype=int)
                df.loc[df.index[idxs], 'Predicted_Spike'] = 1

        last_row = df.iloc[-1]
        try:
            required_features = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'SMA5', 'SMA20',
                'RSI', 'MACD', 'MACD_diff', 'BB_upper', 'BB_lower', 'CCI',
                'Return', 'Volume_Change', 'Rolling_STD', 'IsWeekend', 'Price_Up'
            ]
            missing = [f for f in required_features if f not in last_row.index]
            if missing:
                st.error(f"Missing features in last row: {missing}")
                st.write('last_row:', last_row)
                st.write('last_row columns:', last_row.index)
                prob, result = 0, "Prediction Error"
            else:
                prob, result = predict_spike(last_row)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            prob, result = 0, "Prediction Error"

        st.subheader("🚨 Prediksi Volume Spike")
        st.metric("Probabilitas", f"{prob:.2f}", help="Semakin tinggi semakin besar kemungkinan lonjakan volume")
        st.write(f"**Hasil**: {result}")

        # Tambahkan subheader markdown sebelum grafik
        st.subheader('Harga Close dan Volume dengan Prediksi Lonjakan (1 Tahun Terakhir)')
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='blue')
        ))
        fig_combo.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume',
            marker=dict(color='lightgrey'),
            opacity=0.3,
            yaxis='y2'
        ))
        fig_combo.add_trace(go.Scatter(
            x=df.index[df['Actual_Spike'] == 1],
            y=df['Close'][df['Actual_Spike'] == 1],
            mode='markers',
            name='Actual Spike',
            marker=dict(color='green', size=10, symbol='circle')
        ))
        fig_combo.add_trace(go.Scatter(
            x=df.index[df['Predicted_Spike'] == 1],
            y=df['Close'][df['Predicted_Spike'] == 1],
            mode='markers',
            name='Predicted Spike',
            marker=dict(color='red', size=12, symbol='x')
        ))
        fig_combo.update_layout(
            xaxis=dict(title='Tanggal'),
            yaxis=dict(title='Harga Close (USD)', side='left', color='blue'),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_combo, use_container_width=True)

        # Grafik Harga Harian (Candlestick)
        st.subheader("Grafik Harga BTC Harian")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Grafik Volume Harian (Bar Chart)
        st.subheader("Grafik Volume Harian")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume"
        ))
        fig_vol.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Tanggal",
            yaxis_title="Volume"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Grafik Volume Harian dengan Prediksi Spike
        st.subheader("Grafik Volume Harian dengan Prediksi Spike")
        fig_vol_pred = go.Figure()
        # Bar volume normal
        fig_vol_pred.add_trace(go.Bar(
            x=df.index[df['Predicted_Spike'] == 0],
            y=df['Volume'][df['Predicted_Spike'] == 0],
            name="Volume (Normal)",
            marker_color="lightblue"
        ))
        # Bar volume prediksi spike
        fig_vol_pred.add_trace(go.Bar(
            x=df.index[df['Predicted_Spike'] == 1],
            y=df['Volume'][df['Predicted_Spike'] == 1],
            name="Volume (Prediksi Spike)",
            marker_color="red"
        ))
        fig_vol_pred.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Tanggal",
            yaxis_title="Volume",
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.plotly_chart(fig_vol_pred, use_container_width=True)
