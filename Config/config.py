import pandas as pd

df = pd.read_csv('final_df.csv')

COLUMNS = ['car_name', 'brand', 'model', 'vehicle_age', 'km_driven', 'seller_type','fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power','seats'] 

car_names = df['car_name'].dropna().unique().tolist()
car_brand = df['brand'].dropna().unique().tolist()
car_model = df['model'].dropna().unique().tolist()
car_seller_type = df['seller_type'].dropna().unique().tolist()
car_fuel_type = df['fuel_type'].dropna().unique().tolist()
car_transmission_type = df['transmission_type'].dropna().unique().tolist()

car_catalog = {
    brand: {
        model: sorted(cars["car_name"].unique().tolist())
        for model, cars in brand_df.groupby("model")
    }
    for brand, brand_df in df.dropna(
        subset=["brand", "model", "car_name"]
    ).groupby("brand")
}