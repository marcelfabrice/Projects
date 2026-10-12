from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Customer(BaseModel):
    tenure: int = Field(..., ge=0, description="months the customer has stayed")
    MonthlyCharges: int = Field(..., ge=0, description="monthly cost the customer pays")

    # neue engineered features berechnen
    @computed_field
    @property
    def high_spender(self) -> Literal[0, 1]:
        return 1 if self.MonthlyCharges > 80 else 0

    @computed_field
    @property
    def is_new_customer(self) -> Literal[0, 1]:
        return 1 if self.tenure < 12 else 0

    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    SeniorCitizen: Literal["Yes", "No"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]


class Prediction(BaseModel):
    churn: bool = Field(..., description="True if the customer is predicted to churn")
    churn_probability: float = Field(..., ge=0, le=1)
    shap_values: dict[str, float] = Field(default_factory=dict)
