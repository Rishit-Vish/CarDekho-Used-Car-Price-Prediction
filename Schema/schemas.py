from pydantic import BaseModel, Field, field_validator
from Config.config import (
    car_brand,
    car_fuel_type,
    car_model,
    car_names,
    car_seller_type,
    car_transmission_type
)

class Features(BaseModel):
    car_name : str = Field(...,min_length=1, description = "Name of the car" )
    brand : str = Field(..., min_length = 1, description = "Name of the car brand")
    model : str = Field(..., min_length=1, description = "Name of the car model")
    vehicle_age: int = Field(..., ge=0, description = "How old vehicle you want (In years)")
    km_driven : int = Field(..., ge=0, description = "How many kms a vehicle has already ran")
    seller_type : str = Field(..., min_length = 1, description = "What seller type", examples = ["Individual", "Dealer", "Trustmark Dealer"])
    fuel_type : str = Field(..., min_length =1, description = "Write the fuel type of vehicle", examples = ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    transmission_type : str = Field(..., min_length =1, description = "What transmissions type you want?", examples = ["Manual", "Automatic"])
    mileage: float = Field(..., ge=0, description ="What Mileage you want")
    engine : int = Field(..., ge=0, description="Engine displacement in CC (cubic centimeters)")
    max_power : float = Field(..., ge=0, description = "Maximum engine power in BHP (brake horsepower)")
    seats : int = Field(..., ge=0, description = "How many seats you want")

    @field_validator("car_name")
    @classmethod
    def car_name_validator(cls, value):
        if value not in car_names: 
            raise ValueError("This car is not currently supported. Please select a car from the available options")
        return value 

    @field_validator("brand")
    @classmethod
    def brand_validator(cls, value):
        if value not in car_brand:
            raise ValueError("This brand is not currently supported. Please select a brand from the available options")
        return value

    @field_validator("model")
    @classmethod
    def model_validator(cls, value):
        if value not in car_model:
            raise ValueError("This model is not currently supported. Please select a model from the available options")
        return value

    @field_validator("seller_type")
    @classmethod
    def seller_type_validator(cls, value):
        if value not in car_seller_type:
            raise ValueError("Seller not found")
        return value

    @field_validator("fuel_type") 
    @classmethod
    def fuel_type_validator(cls, value):
        if value not in car_fuel_type:
            raise ValueError(f"Fuel Type not found, Select from {car_fuel_type}")
        return value 

    @field_validator("transmission_type")
    @classmethod
    def transmission_type_validator(cls, value):
        if value not in car_transmission_type:
            raise ValueError(f"Transmission Type not found, Select from {car_transmission_type}")
        return value 
