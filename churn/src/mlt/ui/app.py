import pandas as pd
import requests
import streamlit as st
from mlt.config import PROJECT_ROOT

API_URL = "http://localhost:8000/predict"
BATCH_API_URL = "http://localhost:8000/predict/batch"
METRICS_PATH = PROJECT_ROOT / "figures/model_comparison/comparison.csv"
ROC_CURVE_PATH = PROJECT_ROOT / "figures/model/auc_roc.png"

INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
INTERNET_SERVICE_OPTIONS = ["Yes", "No", "No internet service"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
CUSTOMER_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "SeniorCitizen",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "DeviceProtection",
    "Contract",
]
CATEGORICAL_COLUMNS = [
    "SeniorCitizen",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "DeviceProtection",
    "Contract",
]

st.title("Telco Churn Prediction")


def sync_internet_features() -> None:
    if st.session_state.internet_service == "No":
        for feature in (
            "online_security",
            "online_backup",
            "tech_support",
            "device_protection",
        ):
            st.session_state[feature] = "No internet service"


def prepare_batch_payload(
    data: pd.DataFrame,
) -> tuple[list[dict] | None, pd.Index | None, str | None]:
    missing_columns = [
        column for column in CUSTOMER_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        return None, None, f"Fehlende Spalten: {', '.join(missing_columns)}"

    customers = data[CUSTOMER_COLUMNS].dropna(how="all").copy()
    if customers.empty:
        return None, None, "Die CSV enthält keine Kundendaten."

    for column in CATEGORICAL_COLUMNS:
        customers[column] = (
            customers[column].astype("string").str.strip().replace("", pd.NA)
        )

    for column in ("tenure", "MonthlyCharges"):
        customers[column] = pd.to_numeric(customers[column], errors="coerce")

    invalid_rows = customers[customers.isna().any(axis=1)]
    if not invalid_rows.empty:
        row_numbers = [str(index + 2) for index in invalid_rows.index[:10]]
        return (
            None,
            None,
            f"Ungültige oder leere Werte in CSV-Zeile(n): {', '.join(row_numbers)}",
        )

    customers["tenure"] = customers["tenure"].astype(int)
    customers["MonthlyCharges"] = customers["MonthlyCharges"].astype(int)

    return customers.to_dict(orient="records"), customers.index, None


prediction_tab, batch_tab, performance_tab = st.tabs(
    ["Risiko-Check", "Batch Prediction", "Modell-Performance"]
)

with prediction_tab:
    left, right = st.columns(2)

    with left:
        tenure = st.slider("Tenure (Monate)", min_value=0, step=1, value=0)
        monthly_charges = st.number_input(
            "Monatliche Kosten", min_value=0, step=1, value=0
        )
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=["No", "Yes"],
            format_func=lambda value: "Ja" if value == "Yes" else "Nein",
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=INTERNET_OPTIONS,
            key="internet_service",
            on_change=sync_internet_features,
        )
        payment_method = st.selectbox("Payment Method", options=PAYMENT_OPTIONS)
        contract = st.selectbox("Contract", options=CONTRACT_OPTIONS)

    no_internet = internet_service == "No"
    if no_internet:
        sync_internet_features()

    with right:
        online_security = st.selectbox(
            "Online Security",
            options=INTERNET_SERVICE_OPTIONS,
            key="online_security",
            disabled=no_internet,
        )
        online_backup = st.selectbox(
            "Online Backup",
            options=INTERNET_SERVICE_OPTIONS,
            key="online_backup",
            disabled=no_internet,
        )
        tech_support = st.selectbox(
            "Tech Support",
            options=INTERNET_SERVICE_OPTIONS,
            key="tech_support",
            disabled=no_internet,
        )
        device_protection = st.selectbox(
            "Device Protection",
            options=INTERNET_SERVICE_OPTIONS,
            key="device_protection",
            disabled=no_internet,
        )

    if "feedback" not in st.session_state:  # Zustand über Reruns merken
        st.session_state.feedback = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    submitted = st.button("Risiko berechnen", type="primary")

    if submitted:
        customer = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "SeniorCitizen": senior_citizen,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "TechSupport": tech_support,
            "DeviceProtection": device_protection,
            "Contract": contract,
        }

        response = requests.post(API_URL, json=customer, timeout=10)
        response.raise_for_status()
        st.session_state.last_prediction = response.json()

    if st.session_state.last_prediction is not None:
        prediction = st.session_state.last_prediction
        probability = prediction["churn_probability"]

        st.metric("Churn-Wahrscheinlichkeit", f"{probability:.1%}")
        if prediction["churn"]:
            st.error("Erhöhtes Kündigungsrisiko")
        else:
            st.success("Kein erhöhtes Kündigungsrisiko")

        shap_values = prediction["shap_values"]
        shap_data = (
            pd.DataFrame(
                {
                    "Feature": shap_values.keys(),
                    "SHAP-Wert": shap_values.values(),
                }
            )
            .assign(abs_shap=lambda data: data["SHAP-Wert"].abs())
            .sort_values("abs_shap", ascending=False)
            .drop(columns="abs_shap")
        )

        st.subheader("SHAP-Werte")
        st.bar_chart(shap_data.set_index("Feature")["SHAP-Wert"])
        st.dataframe(shap_data, use_container_width=True, hide_index=True)

        st.subheader("War diese Einschätzung hilfreich?")
        c1, c2 = st.columns(2)
        if c1.button("Korrekt"):
            st.session_state.feedback.append({"proba": probability, "label": "correct"})
        if c2.button("Falsch"):
            st.session_state.feedback.append({"proba": probability, "label": "wrong"})

    if st.session_state.feedback:
        st.caption(f"{len(st.session_state.feedback)} Feedback(s) gesammelt.")

with batch_tab:
    st.header("Batch Prediction")
    st.caption(f"CSV hochladen mit den Spalten: {', '.join(CUSTOMER_COLUMNS)}")

    uploaded_file = st.file_uploader(
        "CSV-Datei auswählen",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("Vorschau")
        st.dataframe(batch_data.head(20), use_container_width=True)

        payload, result_index, validation_error = prepare_batch_payload(batch_data)
        if validation_error:
            st.error(validation_error)
        elif st.button("Batch Prediction starten", type="primary"):
            response = requests.post(BATCH_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            predictions = pd.DataFrame(response.json())
            results = batch_data.loc[result_index].reset_index(drop=True)
            results["churn"] = predictions["churn"]
            results["churn_probability"] = predictions["churn_probability"]

            st.success(f"{len(results)} Predictions erstellt.")
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Ergebnisse als CSV herunterladen",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

with performance_tab:
    st.header("Modellmetriken")

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        knn_metrics = metrics[metrics["model"] == "Best KNN"].set_index("metric")
        metric_labels = {
            "roc_auc": "ROC AUC",
            "f1": "F1-Score",
            "balanced_accuracy": "Balanced Accuracy",
        }
        metric_columns = st.columns(len(metric_labels))
        for column, (metric, label) in zip(metric_columns, metric_labels.items()):
            if metric in knn_metrics.index:
                mean = knn_metrics.at[metric, "mean"]
                std = knn_metrics.at[metric, "std"]
                column.metric(
                    label, f"{mean:.3f}", help=f"Standardabweichung: {std:.3f}"
                )

    st.subheader("ROC-Kurve des KNN-Modells")
    if ROC_CURVE_PATH.exists():
        st.image(str(ROC_CURVE_PATH), caption="ROC-Kurve aus der Cross Validation")
    else:
        st.warning(f"ROC-Kurve nicht gefunden: {ROC_CURVE_PATH}")
