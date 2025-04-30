import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


model = joblib.load('ET.pkl')


feature_ranges = {
    "30kg ABW": {"type": "numerical", "min": 45.000, "max": 100.000, "default": 70.000},
    "litter size": {"type": "numerical", "min": 0, "max": 35, "default": 15},
    "Season": {
        "type": "categorical",
        "options": {
            "Spring": 1,
            "Summer": 2,
            "Autumn": 3,
            "Winter": 4
        },
        "default": "Summer"
    },
    "Birthweight": {"type": "numerical", "min": 0.0, "max": 4.0, "default": 2.0},
    "Parity": {"type": "categorical", "options": [1, 2, 3, 4, 5, 6, 7], "default": 2},
    "Gender": {
        "type": "categorical",
        "options": {
            "Female": 0,
            "Male": 1
        },
        "default": "Female"
    },
}


st.title("Growth Rate Prediction Model with SHAP Visualization")
st.markdown("<h3 style='text-align: center;'>Northwest A&F University, Wu.Lab. China</h3>", unsafe_allow_html=True)

st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        if isinstance(properties["options"], dict):
            # For options with labels (Season and Gender)
            display_options = list(properties["options"].keys())
            selected_label = st.selectbox(
                label=f"{feature} (Select a value)",
                options=display_options,
                index=display_options.index(properties["default"])
            )
            value = properties["options"][selected_label]
        else:
            # For options without labels (Parity)
            value = st.selectbox(
                label=f"{feature} (Select a value)",
                options=properties["options"],
                index=properties["options"].index(properties["default"])
            )
    feature_values.append(value)


features = np.array([feature_values])


if st.button("Predict"):

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    text = f"Based on feature values, predicted possibility of high growth rate is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    class_index = predicted_class 
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
