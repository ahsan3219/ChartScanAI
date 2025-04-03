# import streamlit as st
# import yfinance as yf
# import mplfinance as mpf
# import pandas as pd
# from PIL import Image
# from datetime import datetime, timedelta
# from io import BytesIO
# from ultralytics import YOLO

# # Replace the relative path to your weight file
# model_path = 'weights/custom_yolov8.pt'

# # Logo URL
# logo_url = "images/chartscan.png"

# # Setting page layout
# st.set_page_config(
#     page_title="ChartScanAI",  # Setting page title
#     page_icon="📊",     # Setting page icon
#     layout="wide",      # Setting layout to wide
#     initial_sidebar_state="expanded",    # Expanding sidebar by default
# )

# # Function to download and plot chart
# def generate_chart(ticker, interval="1d", chunk_size=180, figsize=(18, 6.5), dpi=100):
#     if interval == "1h":
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=730)
#         period = None
#     else:
#         start_date = None
#         end_date = None
#         period = "max"
    
#     # Download data for the ticker
#     data = yf.download(ticker, interval=interval, start=start_date, end=end_date, period=period)
    
#     # Ensure the index is a DatetimeIndex and check if data is not empty
#     if not data.empty:
#         data.index = pd.to_datetime(data.index)
#         # Select only the latest 180 candles
#         data = data.iloc[-chunk_size:]

#         # Plot the chart
#         fig, ax = mpf.plot(data, type="candle", style="yahoo",
#                            title=f"{ticker} Latest {chunk_size} Candles",
#                            axisoff=True,
#                            ylabel="",
#                            ylabel_lower="",
#                            volume=False,
#                            figsize=figsize,
#                            returnfig=True)

#         # Save the plot to a BytesIO object
#         buffer = BytesIO()
#         fig.savefig(buffer, format='png', dpi=dpi)  # Ensure DPI is set here
#         buffer.seek(0)
#         return buffer
#     else:
#         st.error("No data found for the specified ticker and interval.")
#         return None

# # Creating sidebar
# with st.sidebar:
#     # Add a logo to the top of the sidebar
#     st.image(logo_url, use_column_width="auto")
#     st.write("")
#     st.header("Configurations")     # Adding header to sidebar
#     # Section to generate and download chart
#     st.subheader("Generate Chart")
#     ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
#     interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"])
#     chunk_size = 180  # Default chunk size
#     if st.button("Generate Chart"):
#         if ticker:
#             chart_buffer = generate_chart(ticker, interval=interval, chunk_size=chunk_size)
#             if chart_buffer:
#                 st.success(f"Chart generated successfully.")
#                 st.download_button(
#                     label=f"Download {ticker} Chart",
#                     data=chart_buffer,
#                     file_name=f"{ticker}_latest_{chunk_size}_candles.png",
#                     mime="image/png"
#                 )
#                 st.image(chart_buffer, caption=f"{ticker} Chart", use_column_width=True)
#         else:
#             st.error("Please enter a valid ticker symbol.")
#     st.write("")
#     st.subheader("Upload Image for Detection")
#     # Adding file uploader to sidebar for selecting images
#     source_img = st.file_uploader(
#         "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

#     # Model Options
#     confidence = float(st.slider(
#         "Select Model Confidence", 25, 100, 30)) / 100

# # Creating main page heading
# st.title("ChartScanAI")
# st.caption('📈 To use the app, choose one of the following options:')

# st.markdown('''
# **Option 1: Upload Your Own Image**
# 1. **Upload Image:** Use the sidebar to upload a candlestick chart image from your local PC.
# 2. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the uploaded chart.

# **Option 2: Generate and Analyze Chart**
# 1. **Generate Chart:** Provide the ticker symbol and interval in the sidebar to create and download a chart (latest 180 candles).
# 2. **Upload Generated Chart:** Use the sidebar to upload the generated chart image.
# 3. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the generated chart.
# ''')

# # Creating two columns on the main page
# col1, col2 = st.columns(2)

# # Adding image to the first column if image is uploaded
# if source_img:
#     with col1:
#         # Opening the uploaded image
#         uploaded_image = Image.open(source_img)
#         # Adding the uploaded image to the page with a caption
#         st.image(uploaded_image,
#                  caption="Uploaded Image",
#                  use_column_width=True
#                  )

# # Load the model
# try:
#     model = YOLO(model_path)
# except Exception as ex:
#     st.error(
#         f"Unable to load model. Check the specified path: {model_path}")
#     st.error(ex)

# # Perform object detection if the button is clicked
# if st.sidebar.button('Detect Objects'):
#     if source_img:
#         # Re-open the image to reset the file pointer
#         source_img.seek(0)
#         uploaded_image = Image.open(source_img)
        
#         # Perform object detection
#         res = model.predict(uploaded_image, conf=confidence)
#         boxes = res[0].boxes
#         res_plotted = res[0].plot()[:, :, ::-1]
#         with col2:
#             st.image(res_plotted,
#                      caption='Detected Image',
#                      use_column_width=True
#                      )
#             try:
#                 with st.expander("Detection Results"):
#                     for box in boxes:
#                         st.write(box.xywh)
#             except Exception as ex:
#                 st.write("Error displaying detection results.")
#     else:
#         st.error("Please upload an image first.")


import streamlit as st
import yfinance as yf # Keep for stock chart generation if desired
import mplfinance as mpf
import pandas as pd
import pandas_ta as ta # For technical indicators and patterns
import ccxt # For crypto data
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
# from ultralytics import YOLO # Keep if you still want YOLO detection

# --- Configuration ---
# model_path = 'weights/custom_yolov8.pt' # YOLO model path (optional)
logo_url = "images/chartscan.png" # Make sure this path is correct relative to the script location

# --- Constants for Strategy ---
BASE_WEIGHTS = {'RSI': 1.44, 'EMA': 1.60, 'ADX': 1.25, 'OBV': 1.358}
CANDLESTICK_WEIGHT = 1.12
ATR_STOP_MULTIPLIER = 2.0
ATR_PROFIT_MULTIPLIER = 4.0
POSITION_SIZE_THRESHOLD = 5.0
# Define some relevant candlestick patterns pandas_ta recognizes
RELEVANT_PATTERNS = [
    'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLBULLISHENGULFING', 'CDLPIERCING', # Bullish
    'CDLSHOOTINGSTAR', 'CDLHANGINGMAN', 'CDLBEARISHENGULFING', 'CDLDARKCLOUDCOVER', # Bearish
    'CDLDOJI', 'CDLDOJISTAR' # Neutral / Indecision
]

# Setting page layout
st.set_page_config(
    page_title="ChartScanAI & Crypto Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

@st.cache_data(ttl=60*15) # Cache data for 15 minutes
def fetch_crypto_data(exchange_id, symbol, timeframe, limit=200):
    """Fetches historical OHLCV data for a crypto symbol using ccxt."""
    try:
        exchange_class = getattr(ccxt, exchange_id)()
        if not exchange_class.has['fetchOHLCV']:
            st.error(f"Exchange {exchange_id} does not support fetching OHLCV data.")
            return None
        
        # CCXT uses milliseconds since epoch
        since = None # Fetch latest data
        # Fetch slightly more data than needed for indicator calculation stability
        ohlcv = exchange_class.fetch_ohlcv(symbol, timeframe, since=since, limit=limit + 50) 
        
        if not ohlcv:
            st.warning(f"No data returned for {symbol} on {exchange_id} with timeframe {timeframe}.")
            return None

        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        
        # Ensure correct data types
        df = df.astype({'Open':'float', 'High':'float', 'Low':'float', 'Close':'float', 'Volume':'float'})

        # Drop the last row if it's incomplete (current candle) for reliable indicator calc
        # Optional: Depends on whether you want signal on closed candle or live candle
        # if exchange_class.timeframes[timeframe]:
        #     tf_in_ms = exchange_class.parse_timeframe(timeframe) * 1000
        #     if datetime.now().timestamp() * 1000 < df.index[-1].timestamp() + tf_in_ms:
        #          df = df[:-1] # Drop potentially incomplete last candle

        if len(df) < 50: # Need enough data for indicators
            st.warning(f"Insufficient data ({len(df)} candles) for calculations after fetching.")
            return None
            
        return df.iloc[-limit:] # Return the requested number of candles + ensuring enough history

    except ccxt.AuthenticationError:
        st.error(f"Authentication error with {exchange_id}. Check API keys if needed.")
        return None
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error fetching data from {exchange_id} for {symbol}: {e}")
        return None
    except ccxt.NetworkError as e:
        st.error(f"Network error connecting to {exchange_id}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching: {e}")
        return None

@st.cache_data
def calculate_indicators(df):
    """Calculates required technical indicators using pandas_ta."""
    if df is None or df.empty:
        return None
    try:
        # Calculate indicators
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.adx(length=14, append=True) # Calculates ADX_14, DMP_14, DMN_14
        df.ta.obv(append=True)
        df.ta.atr(length=14, append=True)

        # Calculate candlestick patterns
        df.ta.cdl_pattern(name=RELEVANT_PATTERNS, append=True)

        # Clean up column names potentially generated by ta
        df.rename(columns={
            'RSI_14': 'RSI', 
            'EMA_50': 'EMA', 
            'ADX_14': 'ADX',
            'OBV': 'OBV',
            'ATR_14': 'ATR'
            }, inplace=True, errors='ignore') # ignore errors if rename fails

        return df.dropna() # Drop rows with NaN values resulting from indicator calculations
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return None

def get_candlestick_signal(df_latest):
    """Identifies the strongest candlestick pattern signal on the latest candle."""
    pattern_signal = 0 # 0: Neutral/None, 1: Bullish, -1: Bearish
    pattern_name = "None"

    for pattern_col in RELEVANT_PATTERNS:
        if pattern_col in df_latest and df_latest[pattern_col] != 0:
            signal_value = df_latest[pattern_col]
            if signal_value > 0: # Bullish pattern
                pattern_signal = 1
                pattern_name = pattern_col
                break # Take first detected bullish pattern
            elif signal_value < 0: # Bearish pattern
                pattern_signal = -1
                pattern_name = pattern_col
                break # Take first detected bearish pattern
            # Ignore neutral patterns (value 0) for weighting purposes

    return pattern_name, pattern_signal

def calculate_trading_signal(df):
    """Applies the trading strategy logic to the latest data."""
    if df is None or len(df) < 2: # Need at least 2 rows for OBV comparison
        st.error("Insufficient data for strategy calculation.")
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    results = {}

    # --- Step 2: Adapt to Market Conditions ---
    adx_value = latest['ADX'] if 'ADX' in latest and pd.notna(latest['ADX']) else 22 # Default neutral if missing
    results['ADX_Value'] = adx_value
    adjusted_weights = BASE_WEIGHTS.copy()

    if adx_value > 25:
        market_condition = "Trending"
        adjusted_weights['EMA'] *= 1.2
        adjusted_weights['ADX'] *= 1.2
    elif adx_value < 20:
        market_condition = "Range-Bound"
        adjusted_weights['RSI'] *= 1.2
    else:
        market_condition = "Neutral / Weak Trend"
    results['Market_Condition'] = market_condition
    results['Adjusted_Weights'] = adjusted_weights

    # --- Step 3: Define Indicator Signals ---
    signals = {}
    # RSI Signal
    rsi_value = latest['RSI'] if 'RSI' in latest and pd.notna(latest['RSI']) else 50 # Default neutral
    if rsi_value < 30: signals['RSI'] = 1
    elif rsi_value > 70: signals['RSI'] = -1
    else: signals['RSI'] = 0
    results['RSI_Value'] = rsi_value
    
    # EMA Signal
    close_price = latest['Close']
    ema_value = latest['EMA'] if 'EMA' in latest and pd.notna(latest['EMA']) else close_price # Default neutral if missing
    if close_price > ema_value: signals['EMA'] = 1
    elif close_price < ema_value: signals['EMA'] = -1
    else: signals['EMA'] = 0
    results['EMA_Value'] = ema_value
    results['Close_Price'] = close_price

    # ADX Signal (used for weighting, signal is neutral in score)
    signals['ADX'] = 0

    # OBV Signal
    obv_value = latest['OBV'] if 'OBV' in latest and pd.notna(latest['OBV']) else 0
    prev_obv_value = previous['OBV'] if 'OBV' in previous and pd.notna(previous['OBV']) else 0
    if obv_value > prev_obv_value: signals['OBV'] = 1
    elif obv_value < prev_obv_value: signals['OBV'] = -1
    else: signals['OBV'] = 0
    results['OBV_Value'] = obv_value
    results['Prev_OBV_Value'] = prev_obv_value

    results['Indicator_Signals'] = signals

    # --- Step 5 & 6: Calculate Scores and Final Signal ---
    # Base Score
    base_total_score = 0
    for ind in signals:
        base_total_score += adjusted_weights[ind] * signals[ind]
    results['Base_Total_Score'] = base_total_score

    # Candlestick Integration
    pattern_name, pattern_signal = get_candlestick_signal(latest)
    results['Candlestick_Pattern'] = pattern_name
    results['Candlestick_Signal'] = pattern_signal

    candlestick_adjustment = 0
    if pattern_signal == 1 and base_total_score > 0: # Bullish pattern confirms bullish score
        candlestick_adjustment = CANDLESTICK_WEIGHT
    elif pattern_signal == -1 and base_total_score < 0: # Bearish pattern confirms bearish score
        candlestick_adjustment = -CANDLESTICK_WEIGHT # Make score more negative

    results['Candlestick_Adjustment'] = candlestick_adjustment

    # Final Score
    final_total_score = base_total_score + candlestick_adjustment
    results['Final_Total_Score'] = final_total_score

    # Final Signal
    if final_total_score > 0: final_signal = "Long (Buy)"
    elif final_total_score < 0: final_signal = "Short (Sell)"
    else: final_signal = "Neutral (Hold)"
    results['Final_Signal'] = final_signal

    # --- Step 4: Risk Management ---
    atr_value = latest['ATR'] if 'ATR' in latest and pd.notna(latest['ATR']) else 0
    results['ATR_Value'] = atr_value

    # Position Sizing
    if abs(final_total_score) > POSITION_SIZE_THRESHOLD:
        position_size = "Full Position"
    elif abs(final_total_score) > 0:
        position_size = "Half Position"
    else:
        position_size = "No Position"
    results['Position_Size'] = position_size

    # Stop-Loss and Take-Profit
    entry_price = close_price # Use latest close as proxy for entry
    if atr_value > 0:
        if final_signal == "Long (Buy)":
            stop_loss = entry_price - (ATR_STOP_MULTIPLIER * atr_value)
            take_profit = entry_price + (ATR_PROFIT_MULTIPLIER * atr_value)
        elif final_signal == "Short (Sell)":
            stop_loss = entry_price + (ATR_STOP_MULTIPLIER * atr_value)
            take_profit = entry_price - (ATR_PROFIT_MULTIPLIER * atr_value)
        else:
            stop_loss = None
            take_profit = None
    else:
        stop_loss = "N/A (ATR=0)"
        take_profit = "N/A (ATR=0)"

    results['Entry_Price_Proxy'] = entry_price
    results['Stop_Loss'] = stop_loss
    results['Take_Profit'] = take_profit
    
    return results


# --- Streamlit App Layout ---

# Sidebar
with st.sidebar:
    try:
        st.image(logo_url, use_column_width="auto")
    except FileNotFoundError:
        st.warning("Logo image not found at specified path.")
    st.write("")
    st.header("⚙️ Configurations")

    # Section for Crypto Strategy Analysis
    st.subheader("📈 Crypto Strategy Analysis")
    exchanges = ccxt.exchanges
    # Filter for exchanges that likely have fetchOHLCV, you might want to refine this list
    common_exchanges = [ 'binance', 'coinbasepro', 'kraken', 'kucoin', 'bybit', 'okx', 'gateio', 'huobi']
    available_exchanges = [ex for ex in common_exchanges if ex in exchanges]
    
    selected_exchange = st.selectbox("Select Exchange", available_exchanges, index=0)
    crypto_symbol = st.text_input("Enter Crypto Symbol (e.g., BTC/USDT):", "BTC/USDT")
    crypto_timeframe = st.selectbox("Select Timeframe", ['1h', '4h', '1d', '1w'], index=2) # Default '1d'
    
    analyze_button = st.button("Analyze Crypto Signal")
    
    # Keep the chart generation/upload section if needed
    st.write("---")
    st.subheader("📊 Chart Generation (Stocks/Indices)")
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    stock_interval = st.selectbox("Select Stock Interval", ["1d", "1h", "1wk"], key="stock_interval")
    chunk_size = 180 

    if st.button("Generate Stock Chart"):
        if stock_ticker:
            # This uses yfinance - keep if you want separate stock charts
            # You might adapt generate_chart to use ccxt data if you want crypto charts here
            st.warning("Chart generation uses yfinance for stocks. Crypto analysis uses CCXT data.")
            # chart_buffer = generate_chart(stock_ticker, interval=stock_interval, chunk_size=chunk_size) 
            # if chart_buffer:
            #     st.success(f"Stock chart generated successfully.")
            #     st.download_button(...)
            #     st.image(chart_buffer, ...)
            st.info("Stock chart generation placeholder.") # Placeholder
        else:
            st.error("Please enter a valid stock ticker symbol.")

    st.write("---")
    # --- Keep YOLO Section if needed ---
    # st.subheader("🖼️ Chart Pattern Detection (YOLO)")
    # source_img = st.file_uploader(...)
    # confidence = st.slider(...)
    # detect_button = st.button('Detect Objects')


# Main Page
st.title("📈 Crypto Strategy Analyzer")
st.caption("Utilizing CCXT for data, Pandas TA for indicators, and a dynamic weighting strategy.")

# Display Strategy Description Expander
with st.expander("ℹ️ Strategy Details", expanded=False):
    st.markdown("""
    **Enhanced Crypto Trading Strategy**

    This strategy combines technical indicators, dynamic weighting based on market conditions, candlestick pattern confirmation, and risk management specifically tailored for cryptocurrency trading using data fetched via `ccxt`.

    **Available Timeframes:** 1 hour ('1h'), 4 hours ('4h'), 1 day ('1d'), 1 week ('1w').

    **1. Indicators:**
    *   Core: RSI(14), EMA(50), ADX(14), OBV
    *   Risk: ATR(14)

    **2. Market Condition (ADX):**
    *   ADX > 25: Trending (Boost EMA & ADX weight)
    *   ADX < 20: Range-Bound (Boost RSI weight)
    *   20-25: Neutral (Base weights)

    **3. Indicator Signals:**
    *   RSI: +1 if < 30, -1 if > 70, 0 otherwise
    *   EMA: +1 if Close > EMA(50), -1 if Close < EMA(50), 0 otherwise
    *   ADX: 0 (influences weights, not direct signal score)
    *   OBV: +1 if Current > Previous, -1 if Current < Previous, 0 otherwise

    **4. Risk Management:**
    *   Stop-Loss: Entry ± (2 * ATR)
    *   Take-Profit: Entry ± (4 * ATR) (1:2 Risk/Reward)
    *   Position Size: Full if |Score| > 5, Half if 0 < |Score| <= 5

    **5. Candlestick Confirmation:**
    *   Recognized patterns (Hammer, Engulfing, etc.) add a weight of `1.12` (positive or negative) *only if* their direction (bullish/bearish) matches the Base Score's direction.

    **6. Final Signal:**
    *   Calculated from the sum of weighted signals + conditional candlestick adjustment.
    *   Output: Long/Short/Neutral signal, position size, SL/TP levels.
    """)


# --- Analysis Execution and Display ---
if analyze_button:
    if not crypto_symbol:
        st.error("Please enter a Crypto Symbol.")
    else:
        st.info(f"Fetching data for {crypto_symbol} on {selected_exchange} ({crypto_timeframe})...")
        
        # 1. Fetch Data
        crypto_data = fetch_crypto_data(selected_exchange, crypto_symbol, crypto_timeframe)

        if crypto_data is not None and not crypto_data.empty:
            st.success(f"Data fetched successfully. ({len(crypto_data)} candles)")
            
            # 2. Calculate Indicators
            crypto_data_with_indicators = calculate_indicators(crypto_data)

            if crypto_data_with_indicators is not None and not crypto_data_with_indicators.empty:
                st.success("Indicators calculated.")
                
                # Display last 5 rows of data with indicators for verification
                st.write("Latest Data & Indicators:")
                st.dataframe(crypto_data_with_indicators.tail())

                # 3. Calculate Trading Signal
                st.info("Calculating trading signal based on the strategy...")
                signal_results = calculate_trading_signal(crypto_data_with_indicators)

                if signal_results:
                    st.success("Strategy analysis complete!")
                    st.subheader(f"Trading Signal for {crypto_symbol} ({crypto_timeframe})")

                    # Display Results in Columns for better layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(label="Final Signal", value=signal_results['Final_Signal'])
                        st.metric(label="Final Total Score", value=f"{signal_results['Final_Total_Score']:.3f}")
                        st.metric(label="Position Size", value=signal_results['Position_Size'])
                        st.metric(label="Entry Price Proxy (Last Close)", value=f"{signal_results['Entry_Price_Proxy']:.4f}")
                        if signal_results['Stop_Loss'] is not None and isinstance(signal_results['Stop_Loss'], (int, float)):
                             st.metric(label="Stop-Loss", value=f"{signal_results['Stop_Loss']:.4f}")
                        else:
                             st.metric(label="Stop-Loss", value=str(signal_results['Stop_Loss']))
                        if signal_results['Take_Profit'] is not None and isinstance(signal_results['Take_Profit'], (int, float)):
                            st.metric(label="Take-Profit", value=f"{signal_results['Take_Profit']:.4f}")
                        else:
                            st.metric(label="Take-Profit", value=str(signal_results['Take_Profit']))
                            
                    with col2:
                        st.text(f"Market Condition: {signal_results['Market_Condition']} (ADX: {signal_results['ADX_Value']:.2f})")
                        st.text(f"Base Score: {signal_results['Base_Total_Score']:.3f}")
                        st.text(f"Candlestick Pattern: {signal_results['Candlestick_Pattern']}")
                        st.text(f"Candlestick Adjustment: {signal_results['Candlestick_Adjustment']:.2f}")
                        st.text(f"Volatility (ATR): {signal_results['ATR_Value']:.4f}")
                        
                        st.write("Indicator Values & Signals:")
                        # Create a small table/dict for clarity
                        sig_data = {
                            'Indicator': ['RSI', 'EMA', 'ADX', 'OBV'],
                            'Value': [
                                f"{signal_results['RSI_Value']:.2f}", 
                                f"{signal_results['EMA_Value']:.2f}", 
                                f"{signal_results['ADX_Value']:.2f}", 
                                f"{signal_results['OBV_Value']:.0f}" # OBV can be large
                            ],
                            'Signal': [
                                signal_results['Indicator_Signals']['RSI'], 
                                signal_results['Indicator_Signals']['EMA'], 
                                signal_results['Indicator_Signals']['ADX'], 
                                signal_results['Indicator_Signals']['OBV']
                            ],
                             'Weight Used': [
                                f"{signal_results['Adjusted_Weights']['RSI']:.3f}",
                                f"{signal_results['Adjusted_Weights']['EMA']:.3f}",
                                f"{signal_results['Adjusted_Weights']['ADX']:.3f}",
                                f"{signal_results['Adjusted_Weights']['OBV']:.3f}"
                             ]
                        }
                        st.dataframe(pd.DataFrame(sig_data))

                else:
                    st.error("Could not calculate trading signal.")
            else:
                st.error("Failed to calculate indicators. Check data.")
        else:
            st.error(f"Could not fetch data for {crypto_symbol} from {selected_exchange}. Please check the symbol and exchange.")

# --- YOLO Detection Logic (Keep if needed) ---
# try:
#     model = YOLO(model_path) # Load YOLO model if path is valid
#     yolo_enabled = True
# except Exception as ex:
#     st.sidebar.error(f"Unable to load YOLO model: {ex}")
#     yolo_enabled = False

# if yolo_enabled and detect_button: # Check if YOLO button was pressed
#     if source_img:
#         # ... (rest of your YOLO detection code using col1, col2) ...
#         st.info("YOLO Detection part executed (if enabled and button clicked).")
#     else:
#         st.sidebar.error("Please upload an image for YOLO detection first.")

# Footer or other info
st.sidebar.write("---")
st.sidebar.info("App combining Crypto Strategy Analysis and optional Chart Tools.")
