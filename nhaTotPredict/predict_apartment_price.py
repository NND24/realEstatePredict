import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def predict_apartment_price(ward, district, size, rooms, toilets, apartment_type, furnishing_sell,urgent):
    # Load the dataset
    df = pd.read_csv('apartmentDataset.csv')

    # Initialize OneHotEncoder with handle_unknown='ignore' to handle unseen categories
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit encoder on existing categories
    encoded_features = encoder.fit_transform(df[['ward', 'district', 'apartment_type', 'furnishing_sell',"urgent"]])

    # Combine encoded features with numeric features
    X = np.hstack((encoded_features.toarray(), df[['size', 'rooms', 'toilets']].values))
    y = df['price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for the new apartment
    new_apartment = pd.DataFrame({
        'ward': [ward],
        'district': [district],
        'size': [size],
        'rooms': [rooms],
        'toilets': [toilets],
        'apartment_type': [apartment_type],
        'furnishing_sell': [furnishing_sell],
        'urgent': [urgent],
    })

    # Encode the new apartment data with handling for unknown categories
    encoded_new_apartment = encoder.transform(new_apartment[['ward', 'district', 'apartment_type', 'furnishing_sell',"urgent"]]).toarray()
    new_apartment_features = np.hstack((encoded_new_apartment, new_apartment[['size', 'rooms', 'toilets']].values))

    # Predict the price
    predicted_price = model.predict(new_apartment_features)

    return predicted_price[0]
