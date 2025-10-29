import re
import difflib
import pandas as pd
import pulp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_visualizations(df, model, X, X_test, y_test):
    # Bar chart for Disaster Categories
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index, palette='viridis')
    plt.title('Distribution of Disaster Categories')
    plt.xlabel('Count')
    plt.ylabel('Disaster Category')
    plt.tight_layout()
    plt.savefig('disaster_categories_distribution.png')
    plt.close()

    # Histogram for Severity Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['severity_score'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Severity Scores')
    plt.xlabel('Severity Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('severity_scores_distribution.png')
    plt.close()

    # Line plot for Disasters Over Time (by Year)
    plt.figure(figsize=(12, 6))
    df['year'].value_counts().sort_index().plot(kind='line', marker='o')
    plt.title('Number of Disasters Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('disasters_over_time.png')
    plt.close()

    # Box plot for Severity Score by Disaster Category
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Category', y='severity_score', data=df, palette='coolwarm')
    plt.title('Severity Score Distribution by Disaster Category')
    plt.xlabel('Disaster Category')
    plt.ylabel('Severity Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('severity_by_category.png')
    plt.close()

    # Feature Importance from RandomForestRegressor
    if model and X is not None:
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 7))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
        plt.title('Feature Importance from Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    # New plots for Qualitative Analysis
    # Histogram for info_len
    plt.figure(figsize=(10, 6))
    sns.histplot(df['info_len'], bins=30, kde=True, color='purple')
    plt.title('Distribution of Disaster Information Length')
    plt.xlabel('Information Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('info_len_distribution.png')
    plt.close()

    # Disaster Categories by Month (Stacked Bar Chart)
    plt.figure(figsize=(14, 7))
    # Map numerical month to month names for better readability
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    df['month_name'] = df['month'].map(month_names)
    category_month = df.groupby(['month_name', 'Category']).size().unstack(fill_value=0)
    # Reorder months for chronological display
    month_order = [month_names[i] for i in sorted(month_names.keys())]
    category_month = category_month.reindex(month_order, axis=0)
    category_month.plot(kind='bar', stacked=True, colormap='tab20')
    plt.title('Disaster Categories by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Disasters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('disaster_categories_by_month.png')
    plt.close()

    # New plot for Comparative Analysis
    # Disasters by Month
    plt.figure(figsize=(10, 6))
    df['month_name'].value_counts().reindex(month_order).plot(kind='bar', color='orange')
    plt.title('Total Number of Disasters by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Disasters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('disasters_by_month.png')
    plt.close()

    # New plots for Ablation Study
    # Correlation Matrix of Features
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('feature_correlation_matrix.png')
    plt.close()

    # Predicted vs. Actual Severity Scores
    if model and X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Perfect prediction line
        plt.title('Predicted vs. Actual Severity Scores')
        plt.xlabel('Actual Severity Score')
        plt.ylabel('Predicted Severity Score')
        plt.tight_layout()
        plt.savefig('predicted_vs_actual_severity.png')
        plt.close()

    # New plot for Inference and Discussion
    # Average Severity Score per Year
    plt.figure(figsize=(12, 6))
    df.groupby('year')['severity_score'].mean().plot(kind='line', marker='o', color='green')
    plt.title('Average Severity Score Per Year')
    plt.xlabel('Year')
    plt.ylabel('Average Severity Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_severity_per_year.png')
    plt.close()

    print("\nAll visualizations generated and saved as PNG files.")

# ... existing code ...

category_le = LabelEncoder()
df['Category_encoded'] = category_le.fit_transform(df['Category'])

X = df[['Category_encoded','duration_days','year','month','day','info_len']]
y = df['severity_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and the LabelEncoder
import pickle
pickle.dump(model, open('disaster_model.pkl', 'wb'))
pickle.dump(category_le, open('category_label_encoder.pkl', 'wb'))

generate_visualizations(df, model, X, X_test, y_test)

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
    print(" No close location found in dataset Titles. Using only category filter.")
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

