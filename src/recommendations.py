# ============================================================
# recommendations.py
# Health risk levels, early-warning messages and recommendations
# for the six sensors available in AQUAIR (temperature, humidity,
# CO2, TVOC, PM2.5, PM10).
#
# IMPORTANT
#   The thresholds below are ADVISORY levels used only to word the
#   recommendations. They do NOT define the GOOD / MODERATE / POOR
#   class (that comes from the monitor's IAQ score, see common.py).
#   They are project assumptions, not medical limits:
#     * PM2.5 15 and PM10 45 ug/m3 follow the WHO 2021 24-hour
#       guideline levels; the "high" levels are project choices.
#     * CO2 1000 ppm is a widely used ventilation indicator.
#     * TVOC, temperature and humidity levels are project choices.
#   This tool gives general guidance, not medical advice.
# ============================================================

SEVERITY = {"GOOD": 0, "MODERATE": 1, "POOR": 2}

HEALTH_RISK = {
    "GOOD": (
        "Low",
        "Air quality is expected to be acceptable. No special action is needed.",
    ),
    "MODERATE": (
        "Moderate",
        "Air quality may become noticeably worse. People who are sensitive "
        "(for example with asthma or allergies) may feel discomfort.",
    ),
    "POOR": (
        "High",
        "Air quality is expected to be poor. Reduce exposure, ventilate or "
        "filter the air, and avoid long stays in the space if possible.",
    ),
}

ADVISORY = {
    "CO2": {"elevated": 1000.0, "high": 1500.0},        # ppm
    "PM25": {"elevated": 15.0, "high": 35.0},           # ug/m3
    "PM10": {"elevated": 45.0, "high": 100.0},          # ug/m3
    "TVOC": {"elevated": 500.0, "high": 1000.0},        # ppb
    "Temperature": {"low": 18.0, "high": 26.0},         # deg C
    "Humidity": {"low": 30.0, "elevated": 60.0, "high": 70.0},   # %
}

# Rising-trend messages are shown when the 30-minute projected change is
# at least this large.
TREND_ALERT = {"CO2": 100.0, "PM25": 5.0, "PM10": 8.0, "TVOC": 150.0,
               "Humidity": 5.0}


def early_warning(current_class, predicted_class):
    """Return (level, message). level is one of error / warning / success / info."""
    cur = SEVERITY[current_class]
    pred = SEVERITY[predicted_class]

    if pred == 2 and cur < 2:
        return ("error", "Early warning: indoor air quality is predicted to "
                         "become POOR within the next 30 minutes.")
    if pred > cur:
        return ("warning", "Warning: indoor air quality is predicted to "
                           f"worsen from {current_class} to {predicted_class} "
                           "within the next 30 minutes.")
    if pred < cur:
        return ("success", "Improvement expected: air quality is predicted to "
                           f"change from {current_class} to {predicted_class} "
                           "within the next 30 minutes.")
    if pred == 2:
        return ("error", "Air quality is POOR now and is predicted to stay "
                         "POOR for the next 30 minutes.")
    return ("info", f"No change expected: air quality is predicted to stay "
                    f"{predicted_class} for the next 30 minutes.")


def get_recommendations(current_class, predicted_class, readings, trends=None):
    """
    Build an ordered list of recommendations.

    readings : dict with keys CO2, PM25, PM10, TVOC, Temperature, Humidity
    trends   : optional dict of projected 30-minute changes per sensor
    Returns a list of dicts: {"level": "urgent" | "action" | "info", "text": str}
    """
    items = []

    def add(level, text):
        if not any(i["text"] == text for i in items):
            items.append({"level": level, "text": text})

    # 1. Overall message from the prediction
    if predicted_class == "POOR":
        add("urgent", "Increase fresh-air ventilation now if outdoor air is "
                      "suitable, and reduce indoor pollution sources.")
    elif predicted_class == "MODERATE":
        add("action", "Open a window or increase ventilation for a short time "
                      "and keep monitoring the readings.")
    else:
        add("info", "Keep the current ventilation routine; air quality is "
                    "predicted to stay in a good range.")

    # 2. Current readings
    co2 = readings["CO2"]
    if co2 >= ADVISORY["CO2"]["high"]:
        add("urgent", "CO₂ is high. Ventilate immediately and reduce "
                      "occupancy or other CO₂ sources if possible.")
    elif co2 >= ADVISORY["CO2"]["elevated"]:
        add("action", "CO₂ is elevated. Improve fresh-air circulation.")

    pm25 = readings["PM25"]
    if pm25 >= ADVISORY["PM25"]["high"]:
        add("urgent", "PM2.5 is high. Remove smoke, dust or combustion "
                      "sources and use filtration if available.")
    elif pm25 >= ADVISORY["PM25"]["elevated"]:
        add("action", "PM2.5 is above the WHO 24-hour guideline level. Check "
                      "for dust or smoke sources.")

    pm10 = readings["PM10"]
    if pm10 >= ADVISORY["PM10"]["high"]:
        add("urgent", "PM10 is high. Check for dust, dry cleaning or "
                      "construction-type sources and clean surfaces.")
    elif pm10 >= ADVISORY["PM10"]["elevated"]:
        add("action", "PM10 is above the WHO 24-hour guideline level. Reduce "
                      "dust generating activities.")

    tvoc = readings["TVOC"]
    if tvoc >= ADVISORY["TVOC"]["high"]:
        add("urgent", "TVOC is high. Check cleaning products, paints, "
                      "solvents, fragrances and other chemical sources, and "
                      "ventilate.")
    elif tvoc >= ADVISORY["TVOC"]["elevated"]:
        add("action", "TVOC is elevated. Reduce unnecessary chemical or "
                      "fragrance sources.")

    temp = readings["Temperature"]
    if temp < ADVISORY["Temperature"]["low"]:
        add("info", "Indoor temperature is low. Keep the space adequately "
                    "warm, but avoid closing ventilation completely.")
    elif temp > ADVISORY["Temperature"]["high"]:
        add("action", "Indoor temperature is high. Improve cooling and "
                      "ventilation.")

    hum = readings["Humidity"]
    if hum >= ADVISORY["Humidity"]["high"]:
        add("action", "Humidity is high, which favours mould and dampness. "
                      "Improve moisture control and ventilation.")
    elif hum >= ADVISORY["Humidity"]["elevated"]:
        add("info", "Humidity is somewhat high. Watch for condensation.")
    elif hum < ADVISORY["Humidity"]["low"]:
        add("info", "Humidity is low. Consider humidification if the air "
                    "feels dry.")

    # 3. Rising trends (early action before the threshold is reached)
    if trends:
        names = {"CO2": "CO₂", "PM25": "PM2.5", "PM10": "PM10",
                 "TVOC": "TVOC", "Humidity": "Humidity"}
        units = {"CO2": "ppm", "PM25": "µg/m³", "PM10": "µg/m³",
                 "TVOC": "ppb", "Humidity": "%"}
        for key, limit in TREND_ALERT.items():
            change = trends.get(key, 0.0)
            if change >= limit:
                add("action",
                    f"{names[key]} is rising (about +{change:.0f} "
                    f"{units[key]} over 30 minutes at the current rate). "
                    "Ventilate early to prevent a further increase.")

    order = {"urgent": 0, "action": 1, "info": 2}
    items.sort(key=lambda i: order[i["level"]])
    return items