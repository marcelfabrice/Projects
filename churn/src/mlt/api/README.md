# Prediction Service for the Model

Train the model and start the FastAPI application:

```shell
dvc repro
uv run uvicorn mlt.api.app:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive Swagger UI or
<http://127.0.0.1:8000/redoc> for the ReDoc documentation.

## Health Check

```shell
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

## Single Prediction

The request body must match the `Customer` schema. The fields `high_spender`
and `is_new_customer` are calculated automatically from `MonthlyCharges` and
`tenure` and must not be supplied.

```shell
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 95,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "TechSupport": "No",
    "SeniorCitizen": "Yes",
    "DeviceProtection": "No",
    "Contract": "Month-to-month"
  }' | python3 -m json.tool
```

The response contains the churn decision and its probability:

```json
{
  "churn": true,
  "churn_probability": 0.75
}
```

## Batch Prediction

The batch endpoint accepts a JSON array of customers and returns a JSON array
of predictions in the same order:

```shell
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "tenure": 2,
      "MonthlyCharges": 95,
      "InternetService": "Fiber optic",
      "PaymentMethod": "Electronic check",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "TechSupport": "No",
      "SeniorCitizen": "Yes",
      "DeviceProtection": "No",
      "Contract": "Month-to-month"
    },
    {
      "tenure": 68,
      "MonthlyCharges": 65,
      "InternetService": "DSL",
      "PaymentMethod": "Bank transfer (automatic)",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "Yes",
      "TechSupport": "Yes",
      "SeniorCitizen": "No",
      "DeviceProtection": "Yes",
      "Contract": "Two year"
    }
  ]' | python3 -m json.tool
```
