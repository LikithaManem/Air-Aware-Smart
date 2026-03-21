import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(page_title="AirAware Smart", layout="wide")

# ------------------------------------------
# CLEAN PROFESSIONAL STYLING
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
}

/* Titles */
h1, h2, h3 {
    color: #333333;
}

/* Reduce vertical gaps */
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
df["date"] = pd.to_datetime(df["date"],dayfirst=True)
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

st.sidebar.dataframe(top_cities, width='stretch', hide_index=True)

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
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

def get_color(aqi):
    if aqi <= 50:
        return "#2e7d32"
    elif aqi <= 100:
        return "#f9a825"
    else:
        return "#c62828"

category = get_category(today_aqi)
color = get_color(today_aqi)

# ------------------------------------------
# KPI SECTION (CLEAN CARDS)
# ------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-box">
        <div style="font-size:25px; color:#666;">City</div>
        <div style="font-size:20px; font-weight:600;">{city}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-box">
        <div style="font-size:25px; color:#666;">AQI Today</div>
        <div style="font-size:22px; font-weight:600; color:{color};">{today_aqi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-box">
        <div style="font-size:25px; color:#666;">Category</div>
        <div style="font-size:20px; font-weight:600; color:{color};">{category}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# ALERT (SUBTLE)
# ------------------------------------------

if today_aqi > 200:
    st.warning("High pollution levels. Avoid outdoor exposure.")
elif today_aqi > 150:
    st.info("Moderate pollution. Consider precautions outdoors.")
else:
    st.success("Air quality is within safe limits.")

# ------------------------------------------
# TABLE + GRAPH
# ------------------------------------------

st.markdown("### Forecast Overview")

col1, col2 = st.columns([1.2, 1.5])

with col1:
    st.markdown("#### Next 7 Days")
    st.dataframe(prediction[["Date", "AQI"]], width='stretch')

with col2:
    st.markdown("#### Trend")
    fig = model.plot(forecast)
    fig.set_size_inches(6, 3)
    st.pyplot(fig)

# ------------------------------------------
# HISTORICAL DATA
# ------------------------------------------

st.markdown("### Historical Data")

hist_display = prophet_df.copy()
hist_display["Date"] = hist_display["ds"].dt.date
hist_display["AQI"] = hist_display["y"]
hist_display = hist_display[["Date", "AQI"]]

st.dataframe(hist_display, width='stretch')
