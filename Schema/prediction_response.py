from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    predicted_selling_price: float = Field(
        ...,
        description = "Estimated resale price in Indian rupees",
        examples = [720840.96]
        )

