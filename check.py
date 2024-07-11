import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    gbc = pickle.load(file)

# Create a dictionary with the provided data
data = {
    'UsingIP': [-1],
    'LongURL': [-1],
    'ShortURL': [1],
    'Symbol@': [1],
    'Redirecting//': [1],
    'PrefixSuffix-': [1],
    'SubDomains': [-1],
    'HTTPS': [-1],
    'DomainRegLen': [0],
    'Favicon': [-1],
    'NonStdPort': [1],
    'HTTPSDomainURL': [1],
    'RequestURL': [1],
    'AnchorURL': [1],
    'LinksInScriptTags': [1],
    'ServerFormHandler': [1],
    'InfoEmail': [1],
    'AbnormalURL': [1],
    'WebsiteForwarding': [1],
    'StatusBarCust': [1],
    'DisableRightClick': [1],
    'UsingPopupWindow': [1],
    'IframeRedirection': [1],
    'AgeofDomain': [1],
    'DNSRecording': [1],
    'WebsiteTraffic': [1],
    'PageRank': [1],
    'GoogleIndex': [1],
    'LinksPointingToPage': [1],
    'StatsReport': [1]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Prepare the input data for prediction
X = np.array(df)  # Convert DataFrame to numpy array

# Make predictions
y_pred = gbc.predict(X)
y_probas = gbc.predict_proba(X)

# Print predictions
print("Predicted class:", y_pred[0])
print("Predicted probabilities:", y_probas[0])
