
import streamlit as st


def show_specifications():
    st.subheader("Steel Specimen Specifications")

    material = st.selectbox(
        "Material",
        ["Steel", "Stainless steel", "Other"],
    )

    col1, col2 = st.columns(2)

    with col1:
        length_mm = st.number_input(
            "Specimen length (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        thickness_mm = st.number_input(
            "Thickness (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )

    with col2:
        width_mm = st.number_input(
            "Specimen width (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        grade = st.text_input(
            "Steel grade (if known)",
            placeholder="Enter documented grade",
        )

    st.caption(
        "Enter actual measured or documented specifications. "
        "These fields do not measure the steel automatically."
    )

    return {
        "material": material,
        "length_mm": length_mm or None,
        "width_mm": width_mm or None,
        "thickness_mm": thickness_mm or None,
        "grade": grade.strip() or None,
    }

