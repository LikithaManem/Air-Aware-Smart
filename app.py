import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="AirAware Smart", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
header {visibility: hidden;}

.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

section.main > div:first-child {
    padding-top: 0rem !important;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}

section[data-testid="stSidebar"] {
    background: #020617;
}

h1 {font-size: 26px !important; color: white;}
h2 {font-size: 20px !important; color: white;}
h3 {font-size: 17px !important; color: white;}
p, label {font-size: 14px !important; color: white;}

.kpi {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}

.status-good {
    background: linear-gradient(90deg,#16a34a,#22c55e);
    padding: 10px;
    border-radius: 8px;
}
.status-mid {
    background: linear-gradient(90deg,#f59e0b,#facc15);
    padding: 10px;
    border-radius: 8px;
}
.status-bad {
    background: linear-gradient(90deg,#dc2626,#ef4444);
    padding: 10px;
    border-radius: 8px;
}

.stButton>button {
    background: #1e293b;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background: #2563eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
df = pd.read_csv("aqi_project_dataset.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎛️ Control Panel")

city = st.sidebar.selectbox("Select City", sorted(df["area"].unique()))

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Explore Insights")

if "view" not in st.session_state:
    st.session_state.view = "dashboard"

if st.sidebar.button("📊 7-Day Forecast"):
    st.session_state.view = "7day"

if st.sidebar.button("📈 Trend Graph"):
    st.session_state.view = "graph"

if st.sidebar.button("📁 Historical Data"):
    st.session_state.view = "history"

if st.sidebar.button("🏭 Top Polluted Cities"):
    st.session_state.view = "top"

if st.sidebar.button("🏠 Dashboard"):
    st.session_state.view = "dashboard"

# ---------------- MODEL ----------------
city_data = df[df["area"] == city]
city_data = city_data.groupby("date")["aqi_value"].mean().reset_index()

prophet_df = city_data.rename(columns={"date": "ds", "aqi_value": "y"})
prophet_df["y"] = prophet_df["y"].clip(lower=10)

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

today = pd.to_datetime("today").normalize()
prediction = forecast[forecast["ds"] >= today].head(8)

prediction["AQI"] = prediction["yhat"].clip(lower=10).round(0).astype(int)
prediction["Date"] = prediction["ds"].dt.strftime("%d-%m-%Y")

today_aqi = int(prediction.iloc[0]["AQI"])

# ---------------- CATEGORY ----------------
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

category = get_category(today_aqi)

view = st.session_state.view

# ================= DASHBOARD =================
if view == "dashboard":

    st.title("🌍 AirAware Smart")
    st.caption("Next-gen Air Quality Intelligence Dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='kpi'>📍 City<br><b>{city}</b></div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='kpi'>🌫 AQI Today<br><b>{today_aqi}</b></div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='kpi'>⚠ Category<br><b>{category}</b></div>", unsafe_allow_html=True)

    if today_aqi <= 50:
        st.markdown("<div class='status-good'>✅ Good air quality</div>", unsafe_allow_html=True)
    elif today_aqi <= 100:
        st.markdown("<div class='status-mid'>⚠ Moderate air quality</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-bad'>🚨 Poor air quality</div>", unsafe_allow_html=True)

    st.markdown("## 📊 Forecast Overview")

    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        st.markdown("### 📅 Next 7 Days")

        prediction["Category"] = prediction["AQI"].apply(get_category)

        table_df = prediction[["Date", "AQI", "Category"]].copy().reset_index(drop=True)
        table_df.insert(0, "No.", range(1, len(table_df) + 1))

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            column_config={
                "No.": st.column_config.NumberColumn("No.", width=60),
                "Date": st.column_config.TextColumn("Date", width=120),
                "AQI": st.column_config.NumberColumn("AQI", width=80),
                "Category": st.column_config.TextColumn("Category", width=300),
            }
        )

    with col2:
        st.markdown("### 📈 Trend Graph")
        fig = model.plot(forecast)
        fig.set_size_inches(7, 3.5)
        st.pyplot(fig)

# ================= OTHER VIEWS =================
elif view == "7day":
    st.title("📊 7-Day Forecast")

    prediction["Category"] = prediction["AQI"].apply(get_category)

    table_df = prediction[["Date", "AQI", "Category"]].copy().reset_index(drop=True)
    table_df.insert(0, "No.", range(1, len(table_df) + 1))

    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        column_config={
            "No.": st.column_config.NumberColumn("No.", width=60),
            "Date": st.column_config.TextColumn("Date", width=120),
            "AQI": st.column_config.NumberColumn("AQI", width=80),
            "Category": st.column_config.TextColumn("Category", width=300),
        }
    )

elif view == "graph":
    st.title("📈 Trend Graph")
    fig = model.plot(forecast)
    fig.set_size_inches(8, 4)
    st.pyplot(fig)

elif view == "history":
    st.title("📁 Historical Data")

    hist = prophet_df.copy()
    hist["Date"] = hist["ds"].dt.strftime("%d-%m-%Y")
    hist["AQI"] = hist["y"].astype(int)

    hist_df = hist[["Date", "AQI"]].copy().reset_index(drop=True)
    hist_df.insert(0, "No.", range(1, len(hist_df) + 1))

    st.dataframe(hist_df, width="stretch", hide_index=True)

elif view == "top":
    st.title("🏭 Top Polluted Cities")

    top = (
        df.groupby("area")["aqi_value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    top = top.rename(columns={"area": "City"})

    top_df = top.copy().reset_index(drop=True)
    top_df.insert(0, "No.", range(1, len(top_df) + 1))

    st.dataframe(top_df, width="stretch", hide_index=True)
