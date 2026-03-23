from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
REQUIRED_COLUMNS = [
    "neighbourhood_cleansed",
    "room_type",
    "accommodates",
    "bedrooms",
    "bathrooms_text",
    "price",
]
CATEGORICAL_COLUMNS = ["neighbourhood_cleansed", "room_type"]
NUMERIC_COLUMNS = ["accommodates", "bedrooms", "bathrooms"]
OPTIONAL_NUMERIC_COLUMNS = [
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "availability_365",
    "latitude",
    "longitude",
]


def resolve_dataset_path(data_dir: Path = DATA_DIR) -> Path:
    preferred_path = data_dir / "listings.csv"
    if preferred_path.is_file():
        return preferred_path

    csv_candidates = sorted(path for path in data_dir.rglob("*.csv") if path.is_file())
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV files were found under '{data_dir}'. "
            "Expected a dataset at 'data/listings.csv' or exactly one CSV somewhere inside 'data/'."
        )

    if len(csv_candidates) > 1:
        candidate_list = "\n".join(f"- {path.relative_to(data_dir)}" for path in csv_candidates)
        raise FileExistsError(
            f"Multiple CSV files were found under '{data_dir}'. "
            "Please keep exactly one dataset there or restore 'data/listings.csv' as a file.\n"
            f"{candidate_list}"
        )

    return csv_candidates[0]


def parse_price(value: object) -> float | None:
    if pd.isna(value):
        return None

    cleaned_value = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned_value:
        return None

    try:
        return float(cleaned_value)
    except ValueError:
        return None


def parse_bathrooms(value: object) -> float | None:
    if pd.isna(value):
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    selected_columns = REQUIRED_COLUMNS + [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in dataframe.columns
    ]
    return dataframe[selected_columns].copy()


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.dropna(subset=REQUIRED_COLUMNS).copy()
    cleaned["price"] = cleaned["price"].apply(parse_price)
    cleaned["bathrooms"] = cleaned["bathrooms_text"].apply(parse_bathrooms)
    cleaned["accommodates"] = pd.to_numeric(cleaned["accommodates"], errors="coerce")
    cleaned["bedrooms"] = pd.to_numeric(cleaned["bedrooms"], errors="coerce")

    available_optional_numeric_columns = [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in cleaned.columns
    ]
    for column in available_optional_numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    numeric_columns = ["price", "bathrooms", "accommodates", "bedrooms"] + available_optional_numeric_columns
    cleaned = cleaned.dropna(subset=numeric_columns).copy()
    cleaned = cleaned[(cleaned["price"] > 10) & (cleaned["price"] < 1000)].copy()

    if cleaned.empty:
        raise ValueError("Cleaning removed all rows. No usable data remains for training.")

    return cleaned


def encode_categorical_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = dataframe.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        label_encoders[column] = encoder

    return encoded, label_encoders


def build_training_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    optional_feature_columns = [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in dataframe.columns
    ]
    feature_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + optional_feature_columns
    features = dataframe[feature_columns].copy()
    target = dataframe["price"].copy()
    return features, target


def train_model(
    features: pd.DataFrame, target: pd.Series
) -> tuple[RandomForestRegressor, float, float]:
    if len(features) < 2:
        raise ValueError("At least 2 cleaned rows are required to split the data and train the model.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return model, r2, rmse


def save_artifacts(
    model: RandomForestRegressor,
    label_encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
    output_path: Path = MODEL_PATH,
) -> None:
    artifact = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
    }
    joblib.dump(artifact, output_path)


def main() -> None:
    dataset_path = resolve_dataset_path()
    raw_dataframe = load_dataset(dataset_path)
    cleaned_dataframe = clean_dataset(raw_dataframe)
    encoded_dataframe, label_encoders = encode_categorical_features(cleaned_dataframe)
    features, target = build_training_data(encoded_dataframe)
    model, r2, rmse = train_model(features, target)
    save_artifacts(model, label_encoders, list(features.columns))

    print(f"Dataset path: {dataset_path}")
    print(f"Rows before cleaning: {len(raw_dataframe)}")
    print(f"Rows after cleaning: {len(cleaned_dataframe)}")
    print(f"R^2 score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved trained model and label encoders to '{MODEL_PATH}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
