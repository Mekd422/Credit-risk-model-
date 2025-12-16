from pydantic import BaseModel, Field
from typing import Optional

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: Optional[float] = None
    transaction_count: int
    avg_transaction_hour: float
    avg_transaction_day: float
    avg_transaction_month: float
    fraud_rate: float
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int

class PredictionResponse(BaseModel):
    risk_probability: float
