from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Berlin Rental Price Predictor", page_icon="🏠", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
ROOM_TYPE_OPTIONS = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
DEFAULT_OPTIONAL_VALUES = {
    "minimum_nights": 2,
    "number_of_reviews": 10,
    "review_scores_rating": 4.5,
    "availability_365": 180,
}
NEIGHBOURHOOD_COORDINATES: dict[str, tuple[float, float]] = {
    "Adlershof": (52.436190, 13.543770),
    "Albrechtstr.": (52.454649, 13.336845),
    "Alexanderplatz": (52.521966, 13.403948),
    "Allende-Viertel": (52.446665, 13.598200),
    "Alt  Treptow": (52.490310, 13.450423),
    "Alt-Hohenschönhausen Nord": (52.557567, 13.499312),
    "Alt-Hohenschönhausen Süd": (52.542968, 13.491387),
    "Alt-Lichtenberg": (52.517616, 13.491336),
    "Altglienicke": (52.412449, 13.538815),
    "Altstadt-Kietz": (52.443511, 13.579815),
    "Barstraße": (52.487346, 13.317004),
    "Baumschulenweg": (52.462536, 13.487127),
    "Biesdorf": (52.504338, 13.558029),
    "Blankenburg/Heinersdorf/Märchenland": (52.577332, 13.441352),
    "Blankenfelde/Niederschönhausen": (52.584874, 13.402547),
    "Bohnsdorf": (52.398811, 13.573335),
    "Britz": (52.452312, 13.437962),
    "Brunnenstr. Nord": (52.543759, 13.376386),
    "Brunnenstr. Süd": (52.533277, 13.395733),
    "Brunsbütteler Damm": (52.534038, 13.148544),
    "Buch": (52.637062, 13.505427),
    "Buchholz": (52.606142, 13.428861),
    "Buckow": (52.419958, 13.422864),
    "Buckow Nord": (52.437771, 13.458749),
    "Charlottenburg Nord": (52.536311, 13.286497),
    "Dammvorstadt": (52.453806, 13.576977),
    "Drakestr.": (52.437458, 13.302638),
    "Düsseldorfer Straße": (52.496967, 13.317933),
    "Falkenhagener Feld": (52.547779, 13.163744),
    "Fennpfuhl": (52.528248, 13.471951),
    "Forst Grunewald": (52.494910, 13.214409),
    "Frankfurter Allee Nord": (52.519227, 13.459863),
    "Frankfurter Allee Süd": (52.509180, 13.482451),
    "Frankfurter Allee Süd FK": (52.507877, 13.462676),
    "Friedenau": (52.470656, 13.335999),
    "Friedrichsfelde Nord": (52.511180, 13.516191),
    "Friedrichsfelde Süd": (52.498919, 13.510815),
    "Friedrichshagen": (52.450252, 13.620101),
    "Gatow / Kladow": (52.469185, 13.141245),
    "Gropiusstadt": (52.426883, 13.461483),
    "Grunewald": (52.488624, 13.285419),
    "Grünau": (52.420694, 13.577282),
    "Hakenfelde": (52.564493, 13.205114),
    "Halensee": (52.498030, 13.296017),
    "Haselhorst": (52.550849, 13.227145),
    "Heerstrasse": (52.511579, 13.238675),
    "Heerstraße Nord": (52.517687, 13.161507),
    "Hellersdorf-Nord": (52.540653, 13.602783),
    "Hellersdorf-Süd": (52.522042, 13.589868),
    "Helmholtzplatz": (52.543699, 13.418487),
    "Johannisthal": (52.443385, 13.505817),
    "Kantstraße": (52.507928, 13.312655),
    "Karl-Marx-Allee-Nord": (52.521468, 13.441269),
    "Karl-Marx-Allee-Süd": (52.511451, 13.441627),
    "Karlshorst": (52.482133, 13.524272),
    "Karow": (52.612285, 13.476741),
    "Kaulsdorf": (52.501785, 13.583650),
    "Kurfürstendamm": (52.502339, 13.320612),
    "Kölln. Vorstadt/Spindlersf.": (52.443545, 13.567419),
    "Köllnische Heide": (52.470494, 13.462046),
    "Köpenick-Nord": (52.465797, 13.580824),
    "Köpenick-Süd": (52.428736, 13.586600),
    "Lankwitz": (52.434263, 13.343020),
    "Lichtenrade": (52.398548, 13.398912),
    "MV 1": (52.595276, 13.349917),
    "MV 2": (52.601445, 13.332119),
    "Mahlsdorf": (52.509812, 13.621710),
    "Malchow, Wartenberg und Falkenberg": (52.576464, 13.541228),
    "Mariendorf": (52.441500, 13.386942),
    "Marienfelde": (52.417929, 13.368382),
    "Marzahn-Mitte": (52.556105, 13.562213),
    "Marzahn-Süd": (52.532913, 13.544432),
    "Mierendorffplatz": (52.525906, 13.305964),
    "Moabit Ost": (52.528231, 13.353591),
    "Moabit West": (52.527931, 13.333888),
    "Müggelheim": (52.414772, 13.666682),
    "Neu Lichtenberg": (52.504307, 13.491706),
    "Neu-Hohenschönhausen Süd": (52.570635, 13.500710),
    "Neue Kantstraße": (52.505487, 13.297200),
    "Neuköllner Mitte/Zentrum": (52.475003, 13.432761),
    "Niederschöneweide": (52.453820, 13.520959),
    "Nord 1": (52.627879, 13.296890),
    "Nord 2": (52.599752, 13.326786),
    "Oberschöneweide": (52.462465, 13.518517),
    "Osloer Straße": (52.557907, 13.386042),
    "Ost 1": (52.566988, 13.371019),
    "Ost 2": (52.567193, 13.353353),
    "Ostpreußendamm": (52.425399, 13.323805),
    "Otto-Suhr-Allee": (52.516295, 13.316085),
    "Pankow Süd": (52.559097, 13.417582),
    "Pankow Zentrum": (52.569180, 13.405527),
    "Parkviertel": (52.551158, 13.347582),
    "Plänterwald": (52.483724, 13.468811),
    "Prenzlauer Berg Nord": (52.549335, 13.422334),
    "Prenzlauer Berg Nordwest": (52.549387, 13.407679),
    "Prenzlauer Berg Ost": (52.531209, 13.448074),
    "Prenzlauer Berg Süd": (52.533343, 13.427943),
    "Prenzlauer Berg Südwest": (52.535274, 13.412943),
    "Rahnsdorf/Hessenwinkel": (52.436149, 13.702171),
    "Regierungsviertel": (52.511860, 13.393450),
    "Reuterstraße": (52.487689, 13.432192),
    "Rixdorf": (52.477412, 13.447021),
    "Rudow": (52.422321, 13.492524),
    "Rummelsburger Bucht": (52.496018, 13.483246),
    "Schillerpromenade": (52.474867, 13.424049),
    "Schloß Charlottenburg": (52.514940, 13.292218),
    "Schloßstr.": (52.461261, 13.319385),
    "Schmargendorf": (52.476809, 13.291941),
    "Schmöckwitz/Karolinenhof/Rauchfangswerder": (52.375232, 13.646515),
    "Schöneberg-Nord": (52.496260, 13.353321),
    "Schöneberg-Süd": (52.484822, 13.353050),
    "Schönholz/Wilhelmsruh/Rosenthal": (52.581995, 13.378824),
    "Siemensstadt": (52.539205, 13.266335),
    "Spandau Mitte": (52.545183, 13.209498),
    "Südliche Friedrichstadt": (52.502538, 13.396018),
    "Teltower Damm": (52.422770, 13.260442),
    "Tempelhof": (52.467364, 13.383350),
    "Tempelhofer Vorstadt": (52.491714, 13.397687),
    "Tiergarten Süd": (52.503468, 13.361749),
    "Volkspark Wilmersdorf": (52.486468, 13.328585),
    "Wedding Zentrum": (52.547575, 13.364528),
    "Weißensee": (52.551731, 13.447546),
    "Weißensee Ost": (52.557931, 13.471437),
    "West 1": (52.569642, 13.276764),
    "West 2": (52.609966, 13.228072),
    "West 3": (52.588492, 13.299013),
    "West 4": (52.570613, 13.316565),
    "West 5": (52.584548, 13.280610),
    "Westend": (52.515208, 13.270586),
    "Wiesbadener Straße": (52.472989, 13.313628),
    "Wilhelmstadt": (52.519607, 13.196106),
    "Zehlendorf  Nord": (52.447973, 13.262845),
    "Zehlendorf  Südwest": (52.423956, 13.179248),
    "nördliche Luisenstadt": (52.501649, 13.427526),
    "südliche Luisenstadt": (52.496431, 13.435230),
}


def ensure_model_artifact(model_path: Path = MODEL_PATH) -> None:
    if model_path.exists():
        return

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    with st.spinner("Training model for the first time, please wait..."):
        from src.train import main as train_main

        train_main()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not created at '{model_path}' after training completed."
        )


@st.cache_resource
def load_model_artifact(model_path: Path = MODEL_PATH) -> dict:
    ensure_model_artifact(model_path)

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary with model metadata.")

    required_keys = {"model", "label_encoders", "feature_columns"}
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        missing_list = ", ".join(sorted(missing_keys))
        raise KeyError(f"Model artifact is missing required keys: {missing_list}")

    label_encoders = artifact["label_encoders"]
    feature_columns = artifact["feature_columns"]
    if not isinstance(label_encoders, dict):
        raise TypeError("Model artifact 'label_encoders' must be a dictionary.")
    if not isinstance(feature_columns, list):
        raise TypeError("Model artifact 'feature_columns' must be a list.")

    for column in ("neighbourhood_cleansed", "room_type"):
        if column not in label_encoders:
            raise KeyError(f"Model artifact is missing the '{column}' label encoder.")
        if not hasattr(label_encoders[column], "classes_"):
            raise TypeError(f"Label encoder for '{column}' is missing classes_.")

    return artifact


def validate_neighbourhood_coordinates(neighbourhood_options: list[str]) -> None:
    missing = [name for name in neighbourhood_options if name not in NEIGHBOURHOOD_COORDINATES]
    extra = [name for name in NEIGHBOURHOOD_COORDINATES if name not in neighbourhood_options]

    if missing or extra:
        messages = []
        if missing:
            messages.append(f"missing coordinates for: {', '.join(missing)}")
        if extra:
            messages.append(f"unexpected coordinate entries for: {', '.join(extra)}")
        raise ValueError("Neighbourhood coordinate map is out of sync with the trained model: " + "; ".join(messages))


def encode_label(value: str, encoder, field_name: str) -> int:
    classes = list(encoder.classes_)
    if value not in classes:
        raise ValueError(
            f"'{value}' is not available in the trained encoder for {field_name}. "
            "Please choose a value that exists in the training data."
        )
    return int(encoder.transform([value])[0])


def build_feature_frame(
    feature_columns: list[str],
    label_encoders: dict,
    inputs: dict[str, float | int | str],
) -> pd.DataFrame:
    neighbourhood = str(inputs["neighbourhood_cleansed"])
    if neighbourhood not in NEIGHBOURHOOD_COORDINATES:
        raise ValueError(f"No coordinates are configured for neighbourhood '{neighbourhood}'.")

    latitude, longitude = NEIGHBOURHOOD_COORDINATES[neighbourhood]
    encoded_values = {
        "neighbourhood_cleansed": encode_label(
            neighbourhood,
            label_encoders["neighbourhood_cleansed"],
            "neighbourhood",
        ),
        "room_type": encode_label(
            str(inputs["room_type"]),
            label_encoders["room_type"],
            "room type",
        ),
        "accommodates": int(inputs["accommodates"]),
        "bedrooms": int(inputs["bedrooms"]),
        "bathrooms": float(inputs["bathrooms"]),
        "minimum_nights": int(DEFAULT_OPTIONAL_VALUES["minimum_nights"]),
        "number_of_reviews": int(DEFAULT_OPTIONAL_VALUES["number_of_reviews"]),
        "review_scores_rating": float(DEFAULT_OPTIONAL_VALUES["review_scores_rating"]),
        "availability_365": int(DEFAULT_OPTIONAL_VALUES["availability_365"]),
        "latitude": float(latitude),
        "longitude": float(longitude),
    }

    unknown_features = [column for column in feature_columns if column not in encoded_values]
    if unknown_features:
        unknown_list = ", ".join(unknown_features)
        raise KeyError(f"Model contains unsupported feature columns: {unknown_list}")

    row = {column: encoded_values[column] for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def render_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0D1B2A;
            --panel: #13263A;
            --panel-strong: #172C44;
            --text: #FFFFFF;
            --muted: #A9B4C2;
            --accent: #FFBF00;
            --border: rgba(255, 191, 0, 0.18);
            --shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 191, 0, 0.08), transparent 28%),
                linear-gradient(180deg, #102136 0%, #0D1B2A 60%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 0.5rem 0 1.75rem;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.6rem, 4vw, 4.6rem);
            line-height: 0.98;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--text);
        }

        .hero-title .accent {
            color: var(--accent);
        }

        .hero-subtitle {
            max-width: 760px;
            margin: 1rem 0 0;
            font-size: 1.02rem;
            line-height: 1.7;
            color: var(--muted);
        }

        .app-card {
            background: linear-gradient(180deg, rgba(23, 44, 68, 0.96), rgba(15, 31, 48, 0.98));
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.5rem;
            min-height: 100%;
        }

        .card-eyebrow {
            margin: 0 0 0.55rem;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--accent);
        }

        .card-title {
            margin: 0 0 1.25rem;
            font-size: 1.4rem;
            line-height: 1.2;
            color: var(--text);
        }

        .result-label {
            margin: 0;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .result-price {
            margin: 0.8rem 0 0;
            font-size: clamp(2.7rem, 4vw, 4.5rem);
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: var(--accent);
        }

        .result-footnote {
            margin: 1rem 0 0;
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--muted);
        }

        h1 a,
        h2 a,
        h3 a,
        h4 a,
        [data-testid='stHeaderActionElements'],
        .stMarkdown a[href^='#'] {
            display: none !important;
            visibility: hidden !important;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stMarkdownContainer"] p {
            color: var(--text);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-testid="stNumberInput"] input {
            background: rgba(255, 255, 255, 0.03) !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
            color: var(--text) !important;
            border-radius: 16px !important;
        }

        div[data-baseweb="select"] svg,
        div[data-baseweb="base-input"] svg {
            fill: var(--accent);
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: rgba(255, 255, 255, 0.12);
        }

        div[data-testid="stSlider"] [role="slider"] {
            background: var(--accent);
            border: 2px solid #ffe08a;
            box-shadow: 0 0 0 6px rgba(255, 191, 0, 0.15);
        }

        div[role="radiogroup"] {
            gap: 0.7rem;
            flex-wrap: wrap;
        }

        div[role="radiogroup"] label {
            border: 1px solid rgba(255, 191, 0, 0.28);
            background: rgba(255, 255, 255, 0.03);
            border-radius: 999px;
            padding: 0.65rem 1rem;
            transition: all 0.2s ease;
            min-width: fit-content;
        }

        div[role="radiogroup"] label:hover {
            border-color: rgba(255, 191, 0, 0.6);
            background: rgba(255, 191, 0, 0.08);
        }

        div[role="radiogroup"] label p {
            color: var(--text);
            font-weight: 600;
        }

        div[role="radiogroup"] label svg {
            display: none;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background: rgba(255, 191, 0, 0.16);
            border-color: var(--accent);
            box-shadow: 0 0 0 1px rgba(255, 191, 0, 0.18) inset;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.2rem;
            border: none;
            border-radius: 999px;
            background: linear-gradient(90deg, #FFBF00 0%, #FFD45A 100%);
            color: #0D1B2A;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            box-shadow: 0 14px 30px rgba(255, 191, 0, 0.18);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #FFD45A 0%, #FFBF00 100%);
            color: #0D1B2A;
        }

        div[data-testid="stAlert"] {
            background: rgba(173, 43, 43, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text);
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_section() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1 class="hero-title">
                Berlin Rental<br><span class="accent">Price Predictor</span>
            </h1>
            <p class="hero-subtitle">
                Find out what your Berlin apartment is worth per night &mdash; powered by real Airbnb data and machine learning
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(predicted_price: float | None) -> None:
    if predicted_price is None:
        display_price = "&euro; -- / night"
    else:
        display_price = f"&euro; {predicted_price:,.0f} / night"

    st.markdown(
        f"""
        <div class="app-card">
            <p class="result-label">ESTIMATED VALUE</p>
            <div class="result-price">{display_price}</div>
            <p class="result-footnote">Estimate based on 14,000+ real Berlin listings</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_custom_css()
    render_hero_section()

    try:
        artifact = load_model_artifact()
        neighbourhood_options = list(artifact["label_encoders"]["neighbourhood_cleansed"].classes_)
        validate_neighbourhood_coordinates(neighbourhood_options)
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    model = artifact["model"]
    label_encoders = artifact["label_encoders"]
    feature_columns = artifact["feature_columns"]

    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None

    left_column, right_column = st.columns([1.2, 0.8], gap="large")

    with left_column:
        st.markdown(
            """
            <div class="app-card">
                <p class="card-eyebrow">Listing Details</p>
            """,
            unsafe_allow_html=True,
        )
        neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)
        room_type = st.radio("Room Type", ROOM_TYPE_OPTIONS, horizontal=True)
        accommodates = st.slider("Number of Guests", min_value=1, max_value=16, value=2)
        bedrooms = st.slider("Bedrooms", min_value=0, max_value=10, value=1)
        bathrooms = st.slider("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

        if st.button("Predict Price"):
            user_inputs = {
                "neighbourhood_cleansed": neighbourhood,
                "room_type": room_type,
                "accommodates": accommodates,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
            }

            try:
                features = build_feature_frame(feature_columns, label_encoders, user_inputs)
                st.session_state.predicted_price = float(model.predict(features)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right_column:
        render_result_card(st.session_state.predicted_price)


if __name__ == "__main__":
    main()
