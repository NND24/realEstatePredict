import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def predict_land_price(ward, district, size, land_type, pty_characteristics,urgent):
    # Load the dataset
    df = pd.read_csv('landDataset.csv')

    # Initialize OneHotEncoder with handle_unknown='ignore' to handle unseen categories
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit encoder on existing categories
    encoded_features = encoder.fit_transform(df[['ward', 'district', 'land_type', 'pty_characteristics',"urgent"]])

    # Combine encoded features with numeric features
    X = np.hstack((encoded_features.toarray(), df[['size']].values))
    y = df['price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for the new land
    new_land = pd.DataFrame({
        'ward': [ward],
        'district': [district],
        'size': [size],
        'land_type': [land_type],
        'pty_characteristics': [pty_characteristics],
        'urgent': [urgent],
    })

    # Encode the new land data with handling for unknown categories
    encoded_new_land = encoder.transform(new_land[['ward', 'district', 'land_type', 'pty_characteristics',"urgent"]]).toarray()
    new_land_features = np.hstack((encoded_new_land, new_land[['size']].values))

    # Predict the price
    predicted_price = model.predict(new_land_features)

    return predicted_price[0]
