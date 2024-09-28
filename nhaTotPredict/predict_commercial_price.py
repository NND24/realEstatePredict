import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def predict_commercial_price(ward, district, size, commercial_type, furnishing_sell,urgent):
    # Load the dataset
    df = pd.read_csv('commercialDataset.csv')

    # Initialize OneHotEncoder with handle_unknown='ignore' to handle unseen categories
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit encoder on existing categories
    encoded_features = encoder.fit_transform(df[['ward', 'district', 'commercial_type', 'furnishing_sell',"urgent"]])

    # Combine encoded features with numeric features
    X = np.hstack((encoded_features.toarray(), df[['size']].values))
    y = df['price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for the new commercial
    new_commercial = pd.DataFrame({
        'ward': [ward],
        'district': [district],
        'size': [size],
        'commercial_type': [commercial_type],
        'furnishing_sell': [furnishing_sell],
        'urgent': [urgent],
    })

    # Encode the new commercial data with handling for unknown categories
    encoded_new_commercial = encoder.transform(new_commercial[['ward', 'district', 'commercial_type', 'furnishing_sell',"urgent"]]).toarray()
    new_commercial_features = np.hstack((encoded_new_commercial, new_commercial[['size']].values))

    # Predict the price
    predicted_price = model.predict(new_commercial_features)

    return predicted_price[0]
