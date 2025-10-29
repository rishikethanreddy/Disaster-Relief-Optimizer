import re
import difflib
import pandas as pd
import pulp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def get_integer_input(prompt, default=0):
    try:
        return int(input(prompt))
    except:
        print(f"Invalid input. Using default value: {default}")
        return default

def extract_duration(duration_str):
    try:
        return float(re.search(r"(\d+)", duration_str).group(1))
    except:
        return 1.0  

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

def optimize_distribution(total_demand, available_inventory):
    ratios_orig = {"food": 0.4, "water": 0.2, "medical": 0.2, "shelter": 0.1}
    min_total = sum(ratios_orig.values()) * total_demand
    scale = 1.0
    ratios_scaled = ratios_orig.copy()
    
    if min_total > available_inventory:
        scale = available_inventory / min_total
        ratios_scaled = {k: v * scale for k, v in ratios_orig.items()}

    prob = pulp.LpProblem("Resource_Allocation", pulp.LpMaximize)
    food = pulp.LpVariable('food', lowBound=0, cat='Integer')
    water = pulp.LpVariable('water', lowBound=0, cat='Integer')
    medical = pulp.LpVariable('medical', lowBound=0, cat='Integer')
    shelter = pulp.LpVariable('shelter', lowBound=0, cat='Integer')

    prob += food + water + medical + shelter
    prob += food + water + medical + shelter <= available_inventory
    prob += food >= ratios_scaled["food"] * total_demand
    prob += water >= ratios_scaled["water"] * total_demand
    prob += medical >= ratios_scaled["medical"] * total_demand
    prob += shelter >= ratios_scaled["shelter"] * total_demand
    prob.solve()

    result = {
        "Food Kits": max(int(food.value()), 0),
        "Water Bottles": max(int(water.value()), 0),
        "Medical Supplies": max(int(medical.value()), 0),
        "Shelter Kits": max(int(shelter.value()), 0)
    }

    key_map = {"food": "Food Kits", "water": "Water Bottles",
               "medical": "Medical Supplies", "shelter": "Shelter Kits"}

    recommended_extra = {}
    for k, v in ratios_orig.items():
        required = int(v * total_demand)
        result_key = key_map[k]
        recommended_extra[k] = max(0, required - result[result_key])

    return result, recommended_extra

df = pd.read_csv('Natural_Disasters_in_India .csv')  
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['year'] = df['Date'].dt.year.fillna(df['Year']).astype(int)
df['month'] = df['Date'].dt.month.fillna(1).astype(int)
df['day'] = df['Date'].dt.day.fillna(1).astype(int)
df['duration_days'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df['duration_days'] = df['duration_days'].fillna(df['duration_days'].median())
df['info_len'] = df['Disaster_Info'].astype(str).str.len()
df['Category'] = df['Title'].apply(categorize_disaster_fuzzy)
np.random.seed(42)
df['severity_score'] = np.random.uniform(0.1, 0.9, size=len(df))

category_le = LabelEncoder()
df['Category_encoded'] = category_le.fit_transform(df['Category'])

X = df[['Category_encoded','duration_days','year','month','day','info_len']]
y = df['severity_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n--- Disaster Relief Input ---")
disaster_title = input("Enter Disaster Title (e.g., 2006 Surat flood): ").strip()
duration_input = input("Enter Duration (in days): ").strip()
#disaster_info = input("Enter Disaster Info (short description): ").strip()
disaster_location = input("Enter Disaster Location: ").strip()
available_inventory = get_integer_input("Enter available total inventory: ", default=1000)
severity_input = input("Enter Severity Score (0-100) or leave blank to predict: ").strip()

category = categorize_disaster_fuzzy(disaster_title)
if category in category_le.classes_:
    category_encoded = category_le.transform([category])[0]
else:
    category_encoded = 0
    print(f"Category '{category}' not found in encoder. Using default encoding 0.")

title_series = df['Title'].astype(str)

matches = [t for t in title_series if disaster_location.lower() in t.lower()]

if not matches:
    close_match = difflib.get_close_matches(disaster_location.lower(),
                                            [t.lower() for t in title_series],
                                            n=1, cutoff=0.6)
    if close_match:
        matched_title = next(t for t in title_series if t.lower() == close_match[0])
        print(f"Matched Location in Dataset Title: {matched_title}")
        matches = [matched_title]

if matches:
    filtered_df = df[title_series.isin(matches) & (df['Category'] == category)]
else:
    print("⚠ No close location found in dataset Titles. Using only category filter.")
    filtered_df = df[df['Category'] == category]

if severity_input == "":
    if not filtered_df.empty:
        severity = filtered_df['severity_score'].mean()
        severity_percent = round(severity * 100, 2)
        print(f"Predicted Severity Score from dataset: {severity_percent}%")
    else:

        duration_days = extract_duration(duration_input)
        info_len = len(disaster_info)
        X_new = pd.DataFrame([[category_encoded, duration_days, 2025, 1, 1, info_len]],
                             columns=['Category_encoded','duration_days','year','month','day','info_len'])
        severity = model.predict(X_new)[0]
        severity = min(max(severity,0),1)
        severity_percent = round(severity*100,2)
        print(f"Predicted Severity Score by model: {severity_percent}%")
else:
    try:
        val = float(severity_input)
        if val < 0 or val > 100:
            print("Value must be between 0-100. Using default 50%.")
            val = 50
    except:
        print("Invalid input. Using default 50%.")
        val = 50
    severity_percent = val
    severity = severity_percent / 100


total_demand = int(severity * 10000)

plan, recommended_extra = optimize_distribution(total_demand, available_inventory)

print("\n--- Disaster Relief Plan ---")
print("Disaster Title:", disaster_title)
print("Disaster Category:", category)
print("Severity Score:", f"{round(severity_percent,2)}%")
print("Predicted Total Demand:", total_demand)
print("Available Inventory Used:", available_inventory)
print("Distribution Plan:", plan)

print("\n Recommended Additional Units Needed per Category:")
print({k.title(): v for k,v in recommended_extra.items()})

