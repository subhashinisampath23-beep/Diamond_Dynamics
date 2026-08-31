
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.title("Diamond Market Segmentation")

project_dir = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# LOAD CLUSTERING MODEL
# ------------------------------------------------------------

cluster_model_path = (
    project_dir /
    "diamond_cluster_model.pkl"
)

cluster_names_path = (
    project_dir /
    "cluster_names.pkl"
)

cluster_objects = joblib.load(
    cluster_model_path
)

cluster_names = joblib.load(
    cluster_names_path
)


# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

st.subheader(
    "Enter Diamond Characteristics"
)

col1, col2 = st.columns(2)

with col1:

    carat = st.number_input(
        "Carat",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.01
    )

    depth = st.number_input(
        "Depth (%)",
        min_value=40.0,
        max_value=80.0,
        value=61.5,
        step=0.1
    )

    table = st.number_input(
        "Table (%)",
        min_value=40.0,
        max_value=80.0,
        value=57.0,
        step=0.1
    )

    x = st.number_input(
        "Length X (mm)",
        min_value=0.1,
        max_value=15.0,
        value=5.5,
        step=0.01
    )


with col2:

    y = st.number_input(
        "Width Y (mm)",
        min_value=0.1,
        max_value=15.0,
        value=5.5,
        step=0.01
    )

    z = st.number_input(
        "Depth Z (mm)",
        min_value=0.1,
        max_value=15.0,
        value=3.4,
        step=0.01
    )

    cut = st.selectbox(
        "Cut",
        [
            "Fair",
            "Good",
            "Very Good",
            "Premium",
            "Ideal"
        ]
    )

    color = st.selectbox(
        "Color",
        [
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J"
        ]
    )


clarity = st.selectbox(
    "Clarity",
    [
        "I1",
        "SI2",
        "SI1",
        "VS2",
        "VS1",
        "VVS2",
        "VVS1",
        "IF"
    ]
)


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

volume = x * y * z

dimension_ratio = (
    (x + y) /
    (2 * z)
)


# ------------------------------------------------------------
# CREATE INPUT DATAFRAME
# ------------------------------------------------------------

input_data = pd.DataFrame([{

    "carat": carat,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": volume,
    "dimension_ratio": dimension_ratio,
    "cut": cut,
    "color": color,
    "clarity": clarity

}])


# ------------------------------------------------------------
# PREDICT CLUSTER
# ------------------------------------------------------------

if st.button(
    "Predict Market Segment",
    type="primary"
):

    try:

        # Apply the exact same preprocessing
        # used during training
        transformed = (
            cluster_objects["preprocessor"]
            .transform(
                input_data
            )
        )

        # Apply the same PCA transformation
        transformed_pca = (
            cluster_objects["pca"]
            .transform(
                transformed
            )
        )

        # Predict cluster
        cluster = (
            cluster_objects["kmeans"]
            .predict(
                transformed_pca
            )[0]
        )

        cluster_number = int(
            cluster
        )

        cluster_name = cluster_names.get(
            cluster_number,
            f"Cluster {cluster_number}"
        )

        st.success(
            f"Market Segment: {cluster_name}"
        )

        st.metric(
            "Cluster Number",
            cluster_number
        )

        st.write(
            f"Carat: {carat:.2f}"
        )

        st.write(
            f"Cut: {cut}"
        )

        st.write(
            f"Color: {color}"
        )

        st.write(
            f"Clarity: {clarity}"
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
