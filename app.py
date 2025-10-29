from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import re
import difflib

app = Flask(__name__)

# Load the trained model and LabelEncoder
try:
    model = pickle.load(open('disaster_model.pkl', 'rb'))
    category_le = pickle.load(open('category_label_encoder.pkl', 'rb'))

except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None
    category_le = None

disaster_categories = {
    "Flood": ["flood", "cyclone", "rain", "tsunami", "flash floods", "monsoon"],
    "Earthquake": ["earthquake", "quake", "aftershock", "tremor"],
    "Cyclone": ["cyclone", "storm", "tropical", "hurricane"],
    "Fire": ["fire", "explosion", "blast", "oil spill", "chemical", "building collapse", "hospital fire", "factory fire"],
    "Stampede": ["stampede", "crowd", "temple", "festival", "kumbh", "railway", "bus accident", "train accident", "derailment"],
    "Pandemic": ["pandemic", "swine flu", "covid", "dengue", "plague", "hepatitis", "alcohol poisoning"],
    "Landslide": ["landslide", "avalanche", "dam failure"],
    "Other": []
}

def categorize_disaster_fuzzy(title):
    title_lower = title.lower()
    for category, keywords in disaster_categories.items():
        if any(kw in title_lower for kw in keywords):
            return category
        match = difflib.get_close_matches(title_lower, keywords, n=1, cutoff=0.6)
        if match:
            return category
    return "Other"

def extract_duration(duration_str):
    try:
        return float(re.search(r"(\d+)", duration_str).group(1))
    except:
        return 1.0

def optimize_distribution(total_demand, available_inventory):
    ratios_orig = {"food": 0.4, "water": 0.2, "medical": 0.2, "shelter": 0.1}
    min_total = sum(ratios_orig.values()) * total_demand
    scale = 1.0
    ratios_scaled = ratios_orig.copy()
    
    if min_total > available_inventory:
        scale = available_inventory / min_total
        ratios_scaled = {k: v * scale for k, v in ratios_orig.items()}

    # Using a simplified approach for optimization for web app,
    # as pulp might be an overkill for a simple demo and can be complex to deploy.
    # This part would ideally be replaced with a more robust optimization library
    # or a pre-calculated distribution based on ratios.
    
    plan = {
        "Food Kits": int(ratios_scaled["food"] * total_demand),
        "Water Bottles": int(ratios_scaled["water"] * total_demand),
        "Medical Supplies": int(ratios_scaled["medical"] * total_demand),
        "Shelter Kits": int(ratios_scaled["shelter"] * total_demand)
    }
    
    # Adjust to ensure total doesn't exceed available_inventory
    current_total = sum(plan.values())
    if current_total > available_inventory:
        reduction_factor = available_inventory / current_total
        for key in plan:
            plan[key] = int(plan[key] * reduction_factor)

    recommended_extra = {}
    for k, v in ratios_orig.items():
        required = int(v * total_demand)
        result_key = [key for key in plan if k.title() in key][0] # Find corresponding key in plan
        recommended_extra[k] = max(0, required - plan[result_key])

    return plan, recommended_extra


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or category_le is None:
        return jsonify({'error': 'Model or LabelEncoder not loaded.'}), 500

    data = request.get_json()
    disaster_title = data.get('disaster_title', '')
    duration_input = data.get('duration_input', '1')
    disaster_location = data.get('disaster_location', '')
    available_inventory = int(data.get('available_inventory', 1000))
    severity_input = data.get('severity_input', '')
    
    # For simplicity, using dummy values for year, month, day, info_len
    # In a real app, these would be derived from the input or a more sophisticated feature engineering
    year = 2025
    month = 1
    day = 1
    info_len = len(disaster_title) # Using title length as a proxy for info_len

    category = categorize_disaster_fuzzy(disaster_title)
    if category in category_le.classes_:
        category_encoded = category_le.transform([category])[0]
    else:
        category_encoded = 0 # Default to 0 or handle unknown category appropriately

    duration_days = extract_duration(duration_input)

    if severity_input == "":
        # Predict severity using the model
        X_new = pd.DataFrame([[category_encoded, duration_days, year, month, day, info_len]],
                             columns=['Category_encoded', 'duration_days', 'year', 'month', 'day', 'info_len'])
        severity = model.predict(X_new)[0]
        severity = min(max(severity, 0), 1)
        severity_percent = round(severity * 100, 2)
    else:
        try:
            val = float(severity_input)
            if val < 0 or val > 100:
                val = 50
            severity_percent = val
            severity = severity_percent / 100
        except ValueError:
            severity_percent = 50
            severity = 0.5

    total_demand = int(severity * 10000)
    plan, recommended_extra = optimize_distribution(total_demand, available_inventory)

    response = {
        "disaster_title": disaster_title,
        "disaster_category": category,
        "severity_score": f"{severity_percent}%",
        "predicted_total_demand": total_demand,
        "available_inventory_used": available_inventory,
        "distribution_plan": plan,
        "recommended_additional_units": {k.title(): v for k, v in recommended_extra.items()}
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)