import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(page_title="AirAware Smart", layout="wide")

# ------------------------------------------
# CLEAN PROFESSIONAL STYLING (FINAL FIX)
# ------------------------------------------

st.markdown("""
<style>

/* Layout spacing */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
}

/* Hide default header */
header {visibility: hidden;}

/* Background */
.stApp {
    background-color: #f8f9fb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eef2f7 !important;
}

/* KPI Cards */
.kpi-box {
    padding: 16px;
    border-radius: 10px;
    background-color: #ffffff;
    text-align: center;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.05);
    color: #000000 !important;
}

/* Force text visibility */
.kpi-box div {
    color: #000000 !important;
}

/* Headings fix */
h1, h2, h3 {
    color: #000000 !important;
    font-weight: 700;
}

/* Ensure all text visible */
p, span {
    color: #000000 !important;
}

/* Responsive font */
@media (max-width: 768px) {
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }

    .kpi-box div {
        font-size: 1rem !important;
    }
}

/* Reduce gaps */
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# TITLE
# ------------------------------------------

st.title("AirAware Smart")
st.caption("Air Quality Prediction Dashboard")

# ------------------------------------------
# LOAD DATA
# ------------------------------------------

df = pd.read_csv("aqi_project_dataset.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df[["date", "area", "aqi_value"]]

# ------------------------------------------
# SIDEBAR
# ------------------------------------------

st.sidebar.title("Settings")

city = st.sidebar.selectbox("Select City", sorted(df["area"].unique()))

st.sidebar.markdown("---")

latest_date = df["date"].max()
latest_data = df[df["date"] == latest_date]

st.sidebar.subheader("Highest AQI Cities")

top_cities = (
    latest_data.groupby("area")["aqi_value"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

top_cities.columns = ["City", "AQI"]
top_cities.insert(0, "Rank", range(1, len(top_cities)+1))

st.sidebar.dataframe(top_cities, width="stretch", hide_index=True)

# ------------------------------------------
# DATA PREPARATION
# ------------------------------------------

city_data = df[df["area"] == city]
city_data = city_data.groupby("date")["aqi_value"].mean().reset_index()

prophet_df = city_data.rename(columns={"date": "ds", "aqi_value": "y"})
prophet_df["y"] = prophet_df["y"].clip(lower=10)

# ------------------------------------------
# MODEL
# ------------------------------------------

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# ------------------------------------------
# PREDICTION
# ------------------------------------------

today = pd.to_datetime(datetime.today().date())
prediction = forecast[forecast["ds"] >= today].head(8)

prediction["AQI"] = prediction["yhat"].clip(lower=0).round(0)
prediction["Date"] = prediction["ds"].dt.date

today_aqi = int(prediction.iloc[0]["AQI"])

# ------------------------------------------
# CATEGORY + COLOR
# ------------------------------------------

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

def get_color(aqi):
    if aqi <= 50:
        return "#2e7d32"
    elif aqi <= 100:
        return "#f9a825"
    elif aqi <= 200:
        return "#ef6c00"
    else:
        return "#c62828"

category = get_category(today_aqi)
color = get_color(today_aqi)

# ------------------------------------------
# KPI SECTION
# ------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-box">
        <div>City</div>
        <div style="font-size:1.4rem; font-weight:600;">{city}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-box">
        <div>AQI Today</div>
        <div style="font-size:1.4rem; font-weight:600; color:{color};">{today_aqi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-box">
        <div>Category</div>
        <div style="font-size:1.4rem; font-weight:600; color:{color};">{category}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# ALERT + HEALTH ADVICE
# ------------------------------------------

if today_aqi > 300:
    st.error("Hazardous air quality. Stay indoors. Wear mask. Avoid all outdoor activities.")
elif today_aqi > 200:
    st.error("Very unhealthy air. Wear mask. Avoid outdoor exposure.")
elif today_aqi > 150:
    st.warning("Unhealthy air quality. Wear mask. Limit outdoor activities.")
elif today_aqi > 100:
    st.warning("Unhealthy for sensitive groups. Consider wearing mask.")
elif today_aqi > 50:
    st.info("Moderate air quality. Sensitive people should take precautions.")
else:
    st.success("Good air quality. Safe for outdoor activities.")

# ------------------------------------------
# TABLE + GRAPH
# ------------------------------------------

st.markdown("### Forecast Overview")

col1, col2 = st.columns([1.2, 1.5])

with col1:
    st.markdown("#### Next 7 Days")
    st.dataframe(prediction[["Date", "AQI"]], width="stretch")

with col2:
    st.markdown("#### Trend")
    fig = model.plot(forecast)
    fig.set_size_inches(5, 3)
    st.pyplot(fig)

# ------------------------------------------
# HISTORICAL DATA
# ------------------------------------------

st.markdown("### Historical Data")

hist_display = prophet_df.copy()
hist_display["Date"] = hist_display["ds"].dt.date
hist_display["AQI"] = hist_display["y"]
hist_display = hist_display[["Date", "AQI"]]

st.dataframe(hist_display, width="stretch")
