import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import streamlit.components.v1 as components

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="AirAware Smart", layout="centered")

# ------------------------------------------
# CSS
# ------------------------------------------
st.markdown("""
<style>

header, footer, #MainMenu {visibility:hidden;}

.block-container {
    max-width: 650px;
    margin: auto;
    padding: 25px;
    border: 1px solid #e0e0e0;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
h1,h2,h3,h4{
            color: #000000 !important;
            }
div[data-testid="stExpander"] summary {
    color: #000000 !important;
    font-weight: 600;
}

div[data-testid="stExpander"] {
    color: #000000 !important;
}
div[data-testid="stAlert"] {
    color: #000000 !important;
    font-weight: 500;
}
div[data-testid="stAlert"] * {
    color: #000000 !important;
}

/* EXTRA FIX FOR WARNING (YELLOW BOX) */
div[data-testid="stAlert"] div {
    color: #000000 !important;
}
label {
    color: #000000 !important;
    font-weight: 500;
}

.title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    color: #0d47a1;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 20px;
}

div.stButton > button {
    height: 40px;
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv("aqi_project_dataset.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# ------------------------------------------
# TITLE
# ------------------------------------------
st.markdown('<div class="title">AirAware Smart</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check Air Quality & Predictions</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# INPUT
# ------------------------------------------
col1, col2 = st.columns([4,1])

with col1:
    city = st.selectbox("Select City", sorted(df["area"].unique()))

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    check = st.button("Check AQI", use_container_width=True)

# ------------------------------------------
# AFTER CLICK
# ------------------------------------------
if check:

    city_data = df[df["area"] == city]
    city_data = city_data.groupby("date")["aqi_value"].mean().reset_index()

    prophet_df = city_data.rename(columns={"date": "ds", "aqi_value": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    today = pd.to_datetime(datetime.today().date())
    prediction = forecast[forecast["ds"] >= today].head(8)

    prediction["AQI"] = prediction["yhat"].round(0)
    prediction["Date"] = prediction["ds"].dt.date

    today_aqi = int(prediction.iloc[0]["AQI"])

    # CATEGORY
    def get_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive"
        elif aqi <= 200:
            return "Unhealthy"
        else:
            return "Hazardous"

    category = get_category(today_aqi)

    # ------------------------------------------
    # AQI CARD
    # ------------------------------------------
    components.html(f"""
    <div style="background:#e3f2fd;padding:20px;border-radius:15px;margin-top:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);font-family:sans-serif;">

        <div style="display:flex;justify-content:space-between;">
            <div>
                <div style="color:#555;">AQI</div>
                <div style="font-size:40px;font-weight:bold;color:#0d47a1;">
                    {today_aqi}
                </div>
                <div style="font-weight:600;color:#0d47a1;">
                    {category}
                </div>
            </div>

            <div style="text-align:right;">
                <div style="font-weight:600;">{city}</div>
                <div style="font-size:12px;color:#555;">Today</div>
            </div>
        </div>

        <div style="height:8px;border-radius:10px;
        background:linear-gradient(to right,#2ecc71,#f1c40f,#e67e22,#e74c3c);
        margin-top:15px;position:relative;">
            <div style="position:absolute;top:-4px;
            left:{min((today_aqi/300)*100,100)}%;
            width:14px;height:14px;background:black;
            border-radius:50%;transform:translateX(-50%);"></div>
        </div>

    </div>
    """, height=200)

    # ------------------------------------------
    # HEALTH ADVISORY
    # ------------------------------------------
    st.markdown("#### Health Advisory")

    if today_aqi > 300:
        st.error("Hazardous air quality. Stay indoors. Wear a mask if necessary. Avoid all outdoor activities.")
    elif today_aqi > 200:
        st.error("Very unhealthy air quality. Wear a mask. Avoid outdoor exposure.")
    elif today_aqi > 150:
        st.warning("Unhealthy air quality. Wear a mask and limit outdoor activities.")
    elif today_aqi > 100:
        st.warning("Unhealthy for sensitive groups. Consider wearing a mask and reduce outdoor exertion.")
    elif today_aqi > 50:
        st.info("Moderate air quality. Sensitive individuals should take precautions.")
    else:
        st.success("Good air quality. Safe for outdoor activities.")

    # ------------------------------------------
    # EXPANDERS WITH EMOJIS
    # ------------------------------------------
    st.markdown("### Explore Insights")
    with st.expander("📊 7-Day Forecast"):
        st.dataframe(prediction[["Date","AQI"]], width="stretch")

    with st.expander("📈 Trend Graph"):
        fig = model.plot(forecast)
        fig.set_size_inches(5,3)
        st.pyplot(fig)

    with st.expander("📁 Historical Data"):
        hist = prophet_df.copy()
        hist["Date"] = hist["ds"].dt.date
        hist["AQI"] = hist["y"]
        st.dataframe(hist[["Date","AQI"]], width="stretch")

    with st.expander("🏭 Top Polluted Cities"):
        latest = df[df["date"] == df["date"].max()]
        top = latest.groupby("area")["aqi_value"].mean().sort_values(ascending=False).head(5).reset_index()
        top.columns = ["City","AQI"]
        st.dataframe(top, width="stretch")
