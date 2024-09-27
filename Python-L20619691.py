# Install required packages
!pip install folium pyarrow scikit-learn

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import folium

# Load data for 2022 and 2023
nbi_2022 = pd.read_csv('C:/Users/alviy/Downloads/TX22.txt', delimiter=',')  # Adjust the path if necessary
nbi_2023 = pd.read_csv('C:/Users/alviy/Downloads/TX23.txt', delimiter=',')  # Adjust the path if necessary

# Identify bridges surveyed in 2023 but not in 2022 using STRUCTURE_NUMBER_008
bridges_2022_ids = set(nbi_2022['STRUCTURE_NUMBER_008'])
bridges_2023_ids = set(nbi_2023['STRUCTURE_NUMBER_008'])

new_bridges = bridges_2023_ids - bridges_2022_ids
new_bridges_data = nbi_2023[nbi_2023['STRUCTURE_NUMBER_008'].isin(new_bridges)]

# Create binary target variables based on ratings
new_bridges_data['DECK_RATING_STATUS'] = (new_bridges_data['DECK_COND_058'] < 5).astype(int)

# Prepare features and target for logistic regression
features = [
    'YEAR_BUILT_027',
    'STRUCTURE_LEN_MT_049',
    'TRAFFIC_LANES_ON_028A',
    'ADT_029',
    'DESIGN_LOAD_031'
]  # Example features to use for modeling

X = new_bridges_data[features].dropna()  # Drop rows with NaN values
y = new_bridges_data['DECK_RATING_STATUS'].loc[X.index]  # Align target variable with features

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X)[:, 1]  # Probability of being less than satisfactory
new_bridges_data['P_DECK_LESS_THAN_SATISFACTORY'] = probabilities

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Map Visualization
map_texas = folium.Map(location=[31.9686, -99.9018], zoom_start=6)

for idx, row in new_bridges_data.iterrows():
    color = 'green' if row['P_DECK_LESS_THAN_SATISFACTORY'] < 0.5 else 'red'
    folium.CircleMarker(
        location=[row['LAT_016'], row['LONG_017']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"Prob: {row['P_DECK_LESS_THAN_SATISFACTORY']:.2f}"
    ).add_to(map_texas)

# Save the map to an HTML file
map_texas.save('texas_bridges_map.html')

# Display the map (if running in a notebook)
map_texas