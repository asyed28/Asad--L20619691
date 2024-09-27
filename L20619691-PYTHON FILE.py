#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import folium

# Load the data from the provided paths
tx_2022 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2022.txt", delimiter=',', low_memory=False)
tx_2023 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2023.txt", delimiter=',', low_memory=False)

# Filter relevant columns: Structure Number, Deck, Culvert, Channel condition, and location (Lat, Long)
columns_of_interest = ['STRUCTURE_NUMBER_008', 'DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061', 'LAT_016', 'LONG_017']

tx_2022_filtered = tx_2022.loc[:, columns_of_interest]
tx_2023_filtered = tx_2023.loc[:, columns_of_interest]

# Drop rows with missing data in the condition columns
tx_2022_filtered = tx_2022_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])
tx_2023_filtered = tx_2023_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])

# Find bridges that were sampled in 2023 but not in 2022
tx_2022_bridge_ids = set(tx_2022_filtered['STRUCTURE_NUMBER_008'])
tx_2023_unique_bridges = tx_2023_filtered[~tx_2023_filtered['STRUCTURE_NUMBER_008'].isin(tx_2022_bridge_ids)].copy()

# Categorize the bridge conditions into Satisfactory (1) or Less than Satisfactory (0)
def categorize_condition(condition):
    if condition in ['9', '8', '7', '6', '5']:  # Satisfactory or better
        return 1
    elif condition in ['4', '3', '2', '1', '0']:  # Less than Satisfactory
        return 0
    else:
        return None

# Apply the categorization
tx_2023_unique_bridges.loc[:, 'DECK_COND_CAT'] = tx_2023_unique_bridges['DECK_COND_058'].apply(categorize_condition)
tx_2023_unique_bridges.loc[:, 'CULVERT_COND_CAT'] = tx_2023_unique_bridges['CULVERT_COND_062'].apply(categorize_condition)
tx_2023_unique_bridges.loc[:, 'CHANNEL_COND_CAT'] = tx_2023_unique_bridges['CHANNEL_COND_061'].apply(categorize_condition)

# Drop rows with missing categories
tx_2023_unique_bridges = tx_2023_unique_bridges.dropna(subset=['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT'])

# Logistic Regression to predict less than satisfactory conditions
# Combine all condition categories into a single target variable for logistic regression
tx_2023_unique_bridges['TARGET'] = tx_2023_unique_bridges[['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT']].min(axis=1)

# Define the features (using LAT and LONG as a simple proxy for modeling)
X = tx_2023_unique_bridges[['LAT_016', 'LONG_017']].astype(float)
y = tx_2023_unique_bridges['TARGET'].astype(int)

# Check if the dataset has enough samples for train/test split
if len(tx_2023_unique_bridges) > 1:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Perform logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_reg.predict(X_test)
    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

    # Print classification results
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Visualize the prediction results on a map
    m = folium.Map(location=[tx_2023_unique_bridges['LAT_016'].mean(), tx_2023_unique_bridges['LONG_017'].mean()], zoom_start=7)

    # Add markers for bridges
    for i, row in tx_2023_unique_bridges.iterrows():
        risk_prob = log_reg.predict_proba([[row['LAT_016'], row['LONG_017']]])[0][1]  # Probability of being "Less than satisfactory"

        # Color code based on risk
        color = 'green' if risk_prob < 0.5 else 'red'

        folium.CircleMarker(
            location=(row['LAT_016'], row['LONG_017']),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Risk: {risk_prob:.2f}"
        ).add_to(m)

    # Display the map
    m.save('bridge_risk_map.html')
    m
else:
    print("Not enough samples for train/test split.")


# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import folium

# Load the data from the provided paths
tx_2022 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2022.txt", delimiter=',', low_memory=False)
tx_2023 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2023.txt", delimiter=',', low_memory=False)

# Check the first few rows of both datasets to verify column names and data
print("TX2022 Columns:", tx_2022.columns)
print("TX2023 Columns:", tx_2023.columns)

# Filter relevant columns: Structure Number, Deck, Culvert, Channel condition, and location (Lat, Long)
columns_of_interest = ['STRUCTURE_NUMBER_008', 'DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061', 'LAT_016', 'LONG_017']

# Check if the required columns exist
missing_cols_2022 = [col for col in columns_of_interest if col not in tx_2022.columns]
missing_cols_2023 = [col for col in columns_of_interest if col not in tx_2023.columns]

if missing_cols_2022:
    print(f"Missing columns in TX2022: {missing_cols_2022}")
if missing_cols_2023:
    print(f"Missing columns in TX2023: {missing_cols_2023}")

# Proceed only if all required columns exist
if not missing_cols_2022 and not missing_cols_2023:
    tx_2022_filtered = tx_2022.loc[:, columns_of_interest]
    tx_2023_filtered = tx_2023.loc[:, columns_of_interest]

    # Drop rows with missing data in the condition columns
    tx_2022_filtered = tx_2022_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])
    tx_2023_filtered = tx_2023_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])

    print(f"Rows after filtering TX2022: {len(tx_2022_filtered)}")
    print(f"Rows after filtering TX2023: {len(tx_2023_filtered)}")

    # Find bridges that were sampled in 2023 but not in 2022
    tx_2022_bridge_ids = set(tx_2022_filtered['STRUCTURE_NUMBER_008'])
    tx_2023_unique_bridges = tx_2023_filtered[~tx_2023_filtered['STRUCTURE_NUMBER_008'].isin(tx_2022_bridge_ids)].copy()

    # Check the size of the unique bridge data
    print(f"Rows in unique bridges sampled in 2023 but not in 2022: {len(tx_2023_unique_bridges)}")

    if len(tx_2023_unique_bridges) > 0:
        # Categorize the bridge conditions into Satisfactory (1) or Less than Satisfactory (0)
        def categorize_condition(condition):
            if condition in ['9', '8', '7', '6', '5']:  # Satisfactory or better
                return 1
            elif condition in ['4', '3', '2', '1', '0']:  # Less than Satisfactory
                return 0
            else:
                return None

        # Apply the categorization
        tx_2023_unique_bridges.loc[:, 'DECK_COND_CAT'] = tx_2023_unique_bridges['DECK_COND_058'].apply(categorize_condition)
        tx_2023_unique_bridges.loc[:, 'CULVERT_COND_CAT'] = tx_2023_unique_bridges['CULVERT_COND_062'].apply(categorize_condition)
        tx_2023_unique_bridges.loc[:, 'CHANNEL_COND_CAT'] = tx_2023_unique_bridges['CHANNEL_COND_061'].apply(categorize_condition)

        # Drop rows with missing categories
        tx_2023_unique_bridges = tx_2023_unique_bridges.dropna(subset=['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT'])

        # Logistic Regression to predict less than satisfactory conditions
        # Combine all condition categories into a single target variable for logistic regression
        tx_2023_unique_bridges['TARGET'] = tx_2023_unique_bridges[['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT']].min(axis=1)

        # Define the features (using LAT and LONG as a simple proxy for modeling)
        X = tx_2023_unique_bridges[['LAT_016', 'LONG_017']].astype(float)
        y = tx_2023_unique_bridges['TARGET'].astype(int)

        # Check if there are any samples left
        if len(X) > 0:
            # Perform logistic regression using all available data
            log_reg = LogisticRegression()
            log_reg.fit(X, y)

            # Predict on the entire dataset
            y_pred = log_reg.predict(X)
            y_pred_prob = log_reg.predict_proba(X)[:, 1]

            # Print classification results
            print(classification_report(y, y_pred))

            # Visualize the prediction results on a map
            m = folium.Map(location=[tx_2023_unique_bridges['LAT_016'].mean(), tx_2023_unique_bridges['LONG_017'].mean()], zoom_start=7)

            # Add markers for bridges
            for i, row in tx_2023_unique_bridges.iterrows():
                risk_prob = log_reg.predict_proba([[row['LAT_016'], row['LONG_017']]])[0][1]  # Probability of being "Less than satisfactory"

                # Color code based on risk
                color = 'green' if risk_prob < 0.5 else 'red'

                folium.CircleMarker(
                    location=(row['LAT_016'], row['LONG_017']),
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Risk: {risk_prob:.2f}"
                ).add_to(m)

            # Display the map
            m.save('bridge_risk_map.html')
            m
        else:
            print("No samples available after filtering for logistic regression.")
    else:
        print("No unique bridges in 2023 that were not in 2022.")
else:
    print("Required columns are missing. Please verify the input files.")


# In[5]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import folium

# Load the data from the provided paths
tx_2022 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2022.txt", delimiter=',', low_memory=False)
tx_2023 = pd.read_csv(r"C:\Users\ragul\Downloads\TX2023.txt", delimiter=',', low_memory=False)

# Filter relevant columns: Structure Number, Deck, Culvert, Channel condition, and location (Lat, Long)
columns_of_interest = ['STRUCTURE_NUMBER_008', 'DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061', 'LAT_016', 'LONG_017']

tx_2022_filtered = tx_2022.loc[:, columns_of_interest]
tx_2023_filtered = tx_2023.loc[:, columns_of_interest]

# Drop rows with missing data in the condition columns
tx_2022_filtered = tx_2022_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])
tx_2023_filtered = tx_2023_filtered.dropna(subset=['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061'])

# Find bridges that were sampled in 2023 but not in 2022
tx_2022_bridge_ids = set(tx_2022_filtered['STRUCTURE_NUMBER_008'])
tx_2023_unique_bridges = tx_2023_filtered[~tx_2023_filtered['STRUCTURE_NUMBER_008'].isin(tx_2022_bridge_ids)].copy()

# Check the size of the unique bridge data
print(f"Rows in unique bridges sampled in 2023 but not in 2022: {len(tx_2023_unique_bridges)}")

# Check condition data before categorization
print("Checking condition data before categorization:")
print(tx_2023_unique_bridges[['DECK_COND_058', 'CULVERT_COND_062', 'CHANNEL_COND_061']].describe())

# Categorize the bridge conditions into Satisfactory (1) or Less than Satisfactory (0)
def categorize_condition(condition):
    if condition in ['9', '8', '7', '6', '5']:  # Satisfactory or better
        return 1
    elif condition in ['4', '3', '2', '1', '0']:  # Less than Satisfactory
        return 0
    else:
        return None

# Apply the categorization
tx_2023_unique_bridges.loc[:, 'DECK_COND_CAT'] = tx_2023_unique_bridges['DECK_COND_058'].apply(categorize_condition)
tx_2023_unique_bridges.loc[:, 'CULVERT_COND_CAT'] = tx_2023_unique_bridges['CULVERT_COND_062'].apply(categorize_condition)
tx_2023_unique_bridges.loc[:, 'CHANNEL_COND_CAT'] = tx_2023_unique_bridges['CHANNEL_COND_061'].apply(categorize_condition)

# Check the distribution of categorized data
print("Categorized Condition Data Distribution:")
print(tx_2023_unique_bridges[['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT']].apply(pd.Series.value_counts))

# Drop rows with missing categories
tx_2023_unique_bridges = tx_2023_unique_bridges.dropna(subset=['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT'])

# Logistic Regression to predict less than satisfactory conditions
# Combine all condition categories into a single target variable for logistic regression
if len(tx_2023_unique_bridges) > 0:
    tx_2023_unique_bridges['TARGET'] = tx_2023_unique_bridges[['DECK_COND_CAT', 'CULVERT_COND_CAT', 'CHANNEL_COND_CAT']].min(axis=1)

    # Define the features (using LAT and LONG as a simple proxy for modeling)
    X = tx_2023_unique_bridges[['LAT_016', 'LONG_017']].astype(float)
    y = tx_2023_unique_bridges['TARGET'].astype(int)

    # Check if there are any samples left
    if len(X) > 0:
        # Perform logistic regression using all available data
        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        # Predict on the entire dataset
        y_pred = log_reg.predict(X)
        y_pred_prob = log_reg.predict_proba(X)[:, 1]

        # Print classification results
        print(classification_report(y, y_pred))

        # Visualize the prediction results on a map
        m = folium.Map(location=[tx_2023_unique_bridges['LAT_016'].mean(), tx_2023_unique_bridges['LONG_017'].mean()], zoom_start=7)

        # Add markers for bridges
        for i, row in tx_2023_unique_bridges.iterrows():
            risk_prob = log_reg.predict_proba([[row['LAT_016'], row['LONG_017']]])[0][1]  # Probability of being "Less than satisfactory"

            # Color code based on risk
            color = 'green' if risk_prob < 0.5 else 'red'

            folium.CircleMarker(
                location=(row['LAT_016'], row['LONG_017']),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Risk: {risk_prob:.2f}"
            ).add_to(m)

        # Display the map
        m.save('bridge_risk_map.html')
        m
    else:
        print("No samples available after filtering for logistic regression.")
else:
    print("No unique bridges in 2023 that were not in 2022.")


# In[8]:



import pandas as pd
import numpy as np
import folium
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assuming your filtered data with latitude, longitude, and condition columns is ready

# Sample Data (Replace with your actual DataFrame)
# Replace this with the actual DataFrame loaded earlier
X = pd.DataFrame({
    'LAT_016': [30.2672, 29.7604, 32.7767, np.nan],  # Replace with actual latitude data
    'LONG_017': [-97.7431, -95.3698, -96.7969, -98.4936]  # Replace with actual longitude data
})

# Sample target variable (0 for less than satisfactory, 1 for satisfactory)
y = np.array([0, 1, 1, 0])  # Replace with actual target variable

# 1. Perform Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X.dropna(), y[:len(X.dropna())])  # Fit model using valid lat/long data

# 2. Predict Probabilities
y_pred_prob = log_reg.predict_proba(X.dropna())[:, 1]  # Get the probabilities for "less than satisfactory"

# 3. Handle NaNs in Latitude and Longitude before visualization
X_clean = X.dropna(subset=['LAT_016', 'LONG_017'])

# Ensure that 'y_pred_prob_clean' contains probabilities corresponding to the cleaned data
y_pred_prob_clean = y_pred_prob  # Since we dropped NaNs before training

# 4. Visualize the results on a map
# Create a map centered at the mean location of the bridges
m = folium.Map(location=[X_clean['LAT_016'].mean(), X_clean['LONG_017'].mean()], zoom_start=7)

# Add markers for each bridge
for i, row in X_clean.iterrows():
    risk_prob = y_pred_prob_clean[i]  # Get the predicted risk probability

    # Set color based on risk probability (green for low, red for high)
    color = 'green' if risk_prob < 0.5 else 'red'

    # Create a marker for each bridge
    folium.CircleMarker(
        location=(row['LAT_016'], row['LONG_017']),
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"Risk of failure: {risk_prob:.2f}"  # Add a popup to show the risk score
    ).add_to(m)

# Save the map to an HTML file and display it
m.save('bridge_risk_map.html')

# Display the map in the notebook (if running in Jupyter)
m


# In[ ]:




