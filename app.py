import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


st.set_page_config(
    page_title="Loan Delinquency Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Loan Delinquency Analysis")
st.write(
    "Explore your loan dataset and compare logistic regression models."
)
st.caption("Educational project: model results are not loan approval decisions.")

FEATURES = [
    "term",
    "gender",
    "age",
    "purpose",
    "home_ownership",
    "FICO",
]
TARGET = "isDelinquent"


def evaluate(model, X, y):
    predictions = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Recall": recall_score(y, predictions, zero_division=0),
        "Precision": precision_score(y, predictions, zero_division=0),
        "F1 score": f1_score(y, predictions, zero_division=0),
    }


def make_pipeline(sampler=None):
    # Your notebook treats these six features as categorical.
    encoder = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    steps = [("encoding", encoder)]

    if sampler is not None:
        steps.append(("sampling", sampler))

    steps.append(
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=1,
            ),
        )
    )

    return Pipeline(steps)


uploaded_file = st.file_uploader(
    "Upload Loan_Delinquent_Dataset CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload your CSV file to start.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

data.columns = data.columns.str.strip()

missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in data.columns
]

if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()

st.subheader("Dataset overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(data))
col2.metric("Columns", len(data.columns))
col3.metric("Duplicate rows", int(data.duplicated().sum()))

st.dataframe(data.head(20), use_container_width=True)

with st.expander("Missing values"):
    st.dataframe(
        data.isna().sum().rename("Missing values").to_frame(),
        use_container_width=True,
    )

with st.expander("Summary statistics"):
    st.dataframe(
        data.describe(include="all"),
        use_container_width=True,
    )

st.subheader("Explore a feature")

feature = st.selectbox("Select a feature", FEATURES)

counts = (
    data[feature]
    .fillna("Missing")
    .astype(str)
    .value_counts()
    .rename("Borrowers")
)

st.bar_chart(counts)

st.subheader("Delinquency by feature")

# Validate target before calculating rates or fitting models.
target_values = pd.to_numeric(data[TARGET], errors="coerce")

if target_values.isna().any() or not target_values.isin([0, 1]).all():
    st.error("isDelinquent must contain only 0 and 1, with no missing values.")
    st.stop()

data[TARGET] = target_values.astype(int)

rates = (
    data.assign(
        selected_feature=data[feature].fillna("Missing").astype(str)
    )
    .groupby("selected_feature")[TARGET]
    .mean()
    .mul(100)
    .rename("Delinquency rate (%)")
)

st.bar_chart(rates)
st.caption("Historical associations do not establish cause and effect.")

st.subheader("Train and compare models")
st.write(
    "The data is split into 60% training, 20% validation and 20% test. "
    "Models are compared using validation recall."
)

if st.button("Train models", type="primary"):
    X = data[FEATURES].copy()

    for column in FEATURES:
        X[column] = X[column].fillna("Missing").astype(str)

    y = data[TARGET]

    if y.nunique() != 2 or y.value_counts().min() < 10:
        st.error(
            "Training needs both target classes and at least "
            "10 rows in each class."
        )
        st.stop()

    try:
        with st.spinner("Training models..."):
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=1,
                stratify=y,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.25,
                random_state=1,
                stratify=y_train_val,
            )

            # Keep SMOTE valid for the smallest cross-validation fold.
            minority_count = int(y_train.value_counts().min())

            if minority_count < 3:
                raise ValueError(
                    "Too few minority-class training rows. "
                    "Please upload a larger dataset."
                )

            neighbors = max(1, min(5, minority_count // 2 - 1))

            models = {
                "Logistic Regression": make_pipeline(),
                "Logistic Regression + SMOTE": make_pipeline(
                    SMOTE(
                        random_state=1,
                        k_neighbors=neighbors,
                    )
                ),
                "Regularised Logistic Regression": GridSearchCV(
                    make_pipeline(
                        SMOTE(
                            random_state=1,
                            k_neighbors=neighbors,
                        )
                    ),
                    param_grid={"model__C": [0.1, 0.3, 0.5, 0.7, 1.0]},
                    scoring="recall",
                    cv=3,
                    error_score="raise",
                ),
                "Logistic Regression + Undersampling": make_pipeline(
                    RandomUnderSampler(random_state=1)
                ),
            }

            training_results = {}
            validation_results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)

                training_results[name] = evaluate(
                    model, X_train, y_train
                )
                validation_results[name] = evaluate(
                    model, X_val, y_val
                )

            training_table = pd.DataFrame(training_results).T
            validation_table = pd.DataFrame(validation_results).T

            best_name = validation_table["Recall"].idxmax()
            best_model = models[best_name]
            test_results = evaluate(best_model, X_test, y_test)

        st.success("Training complete!")

        st.write(
            f"Training rows: {len(X_train)} | "
            f"Validation rows: {len(X_val)} | "
            f"Test rows: {len(X_test)}"
        )

        st.subheader("Training performance")
        st.dataframe(
            training_table.style.format("{:.3f}"),
            use_container_width=True,
        )

        st.subheader("Validation performance")
        st.dataframe(
            validation_table.style.format("{:.3f}"),
            use_container_width=True,
        )

        st.subheader("Validation recall comparison")
        st.bar_chart(validation_table["Recall"])

        st.write(f"**Selected model:** {best_name}")
        st.caption(
            "Selected using validation recall; ties use the first model listed."
        )

        st.subheader("Selected model: test performance")

        metric_columns = st.columns(4)

        for column, (name, value) in zip(
            metric_columns, test_results.items()
        ):
            column.metric(name, f"{value:.1%}")

        st.subheader("Test confusion matrix")

        fig, ax = plt.subplots(figsize=(6, 4))

        ConfusionMatrixDisplay.from_estimator(
            best_model,
            X_test,
            y_test,
            labels=[0, 1],
            display_labels=["Not delinquent", "Delinquent"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SMOTE is applied only inside training folds. "
            "It interpolates encoded categories, so synthetic rows may "
            "contain fractional category indicators."
        )

    except Exception as error:
        st.error(f"Training could not finish: {error}")

