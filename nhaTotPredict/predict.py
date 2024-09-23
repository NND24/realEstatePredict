import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def predict_house_price(ward, district, size, rooms, toilets, floors, house_type, furnishing_sell):
    # Load the dataset
    df = pd.read_csv('houseDataset.csv')

    # Initialize OneHotEncoder with handle_unknown='ignore' to handle unseen categories
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit encoder on existing categories
    encoded_features = encoder.fit_transform(df[['ward', 'district', 'house_type', 'furnishing_sell']])

    # Combine encoded features with numeric features
    X = np.hstack((encoded_features.toarray(), df[['size', 'rooms', 'toilets', 'floors']].values))
    y = df['price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for the new house
    new_house = pd.DataFrame({
        'ward': [ward],
        'district': [district],
        'size': [size],
        'rooms': [rooms],
        'toilets': [toilets],
        'floors': [floors],
        'house_type': [house_type],
        'furnishing_sell': [furnishing_sell],
    })

    # Encode the new house data with handling for unknown categories
    encoded_new_house = encoder.transform(new_house[['ward', 'district', 'house_type', 'furnishing_sell']]).toarray()
    new_house_features = np.hstack((encoded_new_house, new_house[['size', 'rooms', 'toilets', 'floors']].values))

    # Predict the price
    predicted_price = model.predict(new_house_features)

    return predicted_price[0]
