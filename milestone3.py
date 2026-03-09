# ==========================================
# AIRAWARE SMART - MILESTONE 3
# AQI Category Classification and Alert System
# ==========================================

import pandas as pd


# ------------------------------------------
# 1. LOAD PREDICTED AQI DATA (from Milestone 2)
# ------------------------------------------

# This file was created after forecasting AQI using Prophet
df = pd.read_csv("predicted_aqi.csv")

print("Predicted AQI Data:")
print(df.head())


# ------------------------------------------
# 2. FUNCTION TO DETERMINE AQI CATEGORY
# ------------------------------------------

def get_aqi_category(aqi_value):

    # Check AQI range and return corresponding category

    if aqi_value <= 50:
        return "Good"

    elif aqi_value <= 100:
        return "Moderate"

    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi_value <= 200:
        return "Unhealthy"

    elif aqi_value <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


# ------------------------------------------
# 3. APPLY CATEGORY FUNCTION
# ------------------------------------------

# Apply the function to predicted AQI column
df["AQI_Category"] = df["yhat"].apply(get_aqi_category)

print("\nAQI with Categories:")
print(df)


# ------------------------------------------
# 4. FUNCTION TO GENERATE HEALTH ALERTS
# ------------------------------------------

def generate_alert(aqi_value):

    # Give warning messages based on AQI level

    if aqi_value > 200:
        return "⚠ HIGH ALERT! Avoid outdoor activities."

    elif aqi_value > 150:
        return "⚠ Warning! Wear mask outside."

    else:
        return "Air quality is safe."


# ------------------------------------------
# 5. APPLY ALERT FUNCTION
# ------------------------------------------

df["Health_Alert"] = df["yhat"].apply(generate_alert)


# ------------------------------------------
# 6. DISPLAY FINAL REPORT
# ------------------------------------------

print("\nFinal AQI Alert Report:")
print(df)


# ------------------------------------------
# 7. SAVE FINAL RESULT
# ------------------------------------------

df.to_csv("final_aqi_alert_report.csv", index=False)

print("\nFinal alert report saved successfully.")