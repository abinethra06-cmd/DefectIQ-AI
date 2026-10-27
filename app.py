
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="DefectIQ AI | Steel Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DASHBOARD STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce4ed;
    padding: 16px;
    border-radius: 12px;
}
.hero {
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #102a43, #176b87);
    color: white;
}
.hero h1 {
    color: white;
}
.hero p {
    color: #e5f3f8;
}
h1, h2, h3 {
    color: #172b4d;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "latest_image_rgb" not in st.session_state:
    st.session_state.latest_image_rgb = None

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.markdown("## 🔍 DefectIQ AI")
st.sidebar.caption("Steel Surface Quality Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Inspect Image",
        "Batch Inspection",
        "Quality Analytics",
        "Reports",
        "Model Health",
    ],
)

st.sidebar.divider()
st.sidebar.caption("YOLO • Computer Vision • Quality Control")
st.sidebar.caption(
    "Prototype decision support. Verify critical results "
    "with a qualified inspector."
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def load_image(uploaded_file):
    """Convert an uploaded image to RGB and OpenCV BGR."""
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def run_inspection(image_bgr, filename):
    """Use the existing Member B inspection pipeline."""
    try:
        from pipeline import inspect

        return inspect(
            image_bgr=image_bgr,
            image_name=filename,
            profile="Standard",
            pixels_per_mm=None,
            specimen=None,
            save=True,
        )

    except Exception as exc:
        st.error("The inspection pipeline could not complete.")
        st.exception(exc)
        return None


def get_defects(result):
    defects = result.get("defects", []) if isinstance(result, dict) else []
    return defects if isinstance(defects, list) else []


def history_dataframe():
    return pd.DataFrame(st.session_state.inspection_history)


def record_history(filename, result):
    defects = get_defects(result)

    st.session_state.inspection_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Decision": str(result.get("decision", "UNKNOWN")).upper(),
        "Severity": float(result.get("max_score", 0) or 0),
        "Defect Count": len(defects),
        "Reason": str(result.get("reason", "")),
        "Inference (ms)": float(result.get("inference_ms", 0) or 0),
    })

    st.session_state.latest_result = result


def show_result(result, original_rgb=None):
    if not result:
        return

    defects = get_defects(result)
    decision = str(result.get("decision", "UNKNOWN")).upper()
    score = float(result.get("max_score", 0) or 0)

    st.subheader("Inspection Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Decision", decision)
    col2.metric("Maximum Severity", f"{score:.1f}/100")
    col3.metric("Defects Detected", len(defects))

    if decision == "ACCEPT":
        st.success("ACCEPT — the configured rules did not escalate this inspection.")
    elif decision == "REWORK":
        st.warning("REWORK — corrective action is recommended by the configured rules.")
    elif decision == "REJECT":
        st.error("REJECT — the configured rejection threshold was reached.")
    elif decision == "MANUAL_REVIEW":
        st.warning("MANUAL REVIEW — a person should review this result.")

    st.write("**Reason:**", result.get("reason", "No reason provided."))

    image_col, details_col = st.columns([1.1, 1])

    with image_col:
        st.markdown("#### Annotated Inspection Image")

        annotated = result.get("annotated")

        if annotated is not None:
            try:
                annotated_rgb = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )
                st.image(
                    annotated_rgb,
                    use_container_width=True,
                )
            except Exception:
                st.info("The annotated image could not be displayed.")
        elif original_rgb is not None:
            st.image(original_rgb, use_container_width=True)
            st.info("The pipeline did not return an annotated image.")

    with details_col:
        st.markdown("#### Defect Details")

        if defects:
            rows = [{
                "Defect": d.get("type", "Unknown"),
                "Confidence": d.get("confidence"),
                "Width (px)": d.get("width_px"),
                "Height (px)": d.get("height_px"),
                "Severity": d.get("score"),
                "Decision": d.get("decision"),
            } for d in defects]

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No defect entries were returned. If unexpected, "
                "review the detector output and confidence threshold."
            )

    with st.expander("Technical Result Details"):
        st.json({
            key: value
            for key, value in result.items()
            if key != "annotated"
        })

    st.caption(
        f"Inference time: "
        f"{float(result.get('inference_ms', 0) or 0):.1f} ms"
    )


def decision_counts(history):
    if history.empty or "Decision" not in history:
        return pd.Series(dtype="int64")

    return history["Decision"].astype(str).str.upper().value_counts()


# ==================================================
# 1. OVERVIEW
# ==================================================
if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>DefectIQ AI</h1>
        <p>
            AI-assisted steel surface inspection,
            defect analysis, and quality reporting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Inspect individual images, process image batches, "
        "analyse quality results, and generate reports."
    )

    history = history_dataframe()
    total = len(history)
    decisions = decision_counts(history)

    accepted = int(decisions.get("ACCEPT", 0))
    needs_attention = sum(
        int(decisions.get(x, 0))
        for x in ["REWORK", "REJECT", "MANUAL_REVIEW"]
    )

    total_defects = (
        int(history["Defect Count"].sum()) if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspections This Session", total)
    c2.metric("Defects Detected", total_defects)
    c3.metric("Accepted", accepted)
    c4.metric("Needs Attention", needs_attention)

    st.divider()
    st.subheader("Inspection Workflow")

    a, b, c = st.columns(3)

    a.markdown("### 01 · Upload")
    a.write("Choose one or multiple steel surface images.")

    b.markdown("### 02 · Detect")
    b.write("Run the existing YOLO inspection pipeline.")

    c.markdown("### 03 · Review")
    c.write("Review defects, severity, decisions, and reports.")

    st.info(
        "Severity thresholds are configurable prototype rules, "
        "not certified industrial acceptance standards."
    )

    if total:
        st.subheader("Recent Inspections")
        st.dataframe(
            history.tail(8).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No inspections recorded in this session yet. "
            "Open Inspect Image to begin."
        )


# ==================================================
# 2. INSPECT IMAGE
# ==================================================
elif page == "Inspect Image":

    st.title("Inspect a Steel Surface")
    st.write(
        "Upload one image to detect defects and view "
        "the inspection decision."
    )

    uploaded = st.file_uploader(
        "Choose a steel surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="single_upload",
    )

    if uploaded:
        rgb, bgr = load_image(uploaded)

        st.image(
            rgb,
            caption=f"Uploaded image: {uploaded.name}",
            use_container_width=True,
        )

        if st.button("Run Inspection", type="primary"):
            with st.spinner("Inspecting image..."):
                result = run_inspection(bgr, uploaded.name)

            if result is not None:
                record_history(uploaded.name, result)
                st.session_state.latest_image_rgb = rgb
                st.success("Inspection completed.")
                show_result(result, rgb)

    elif st.session_state.latest_result is not None:
        st.caption("Latest result from this session")
        show_result(
            st.session_state.latest_result,
            st.session_state.latest_image_rgb,
        )


# ==================================================
# 3. BATCH INSPECTION
# ==================================================
elif page == "Batch Inspection":

    st.title("Batch Inspection")
    st.write(
        "Inspect several steel images and download "
        "a consolidated CSV report."
    )

    files = st.file_uploader(
        "Choose multiple steel surface images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if files and st.button("Inspect All Images", type="primary"):

        rows = []
        progress = st.progress(0)

        for index, uploaded in enumerate(files):
            try:
                _, bgr = load_image(uploaded)

                with st.spinner(f"Inspecting {uploaded.name}..."):
                    result = run_inspection(bgr, uploaded.name)

                if result is not None:
                    row = {
                        "Timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Filename": uploaded.name,
                        "Decision": str(
                            result.get("decision", "UNKNOWN")
                        ).upper(),
                        "Severity": float(
                            result.get("max_score", 0) or 0
                        ),
                        "Defect Count": len(get_defects(result)),
                        "Reason": str(result.get("reason", "")),
                        "Inference (ms)": float(
                            result.get("inference_ms", 0) or 0
                        ),
                    }

                    rows.append(row)
                    st.session_state.inspection_history.append(row)

            except Exception as exc:
                st.error(
                    f"Could not process {uploaded.name}: {exc}"
                )

            progress.progress((index + 1) / len(files))

        if rows:
            batch_df = pd.DataFrame(rows)

            st.subheader("Batch Results")
            st.dataframe(
                batch_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Batch Results (CSV)",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="defectiq_batch_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No images were inspected successfully.")


# ==================================================
# 4. QUALITY ANALYTICS
# ==================================================
elif page == "Quality Analytics":

    st.title("Quality Analytics")
    history = history_dataframe()

    if history.empty:
        st.info(
            "Run inspections first. Analytics use records "
            "from the current browser session."
        )
    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Inspections", len(history))
        c2.metric(
            "Total Defects",
            int(history["Defect Count"].sum()),
        )
        c3.metric(
            "Average Severity",
            f"{history['Severity'].mean():.1f}/100",
        )

        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("Decisions by Category")
            st.bar_chart(decision_counts(history))

        with chart2:
            st.subheader("Defects per Image")
            st.bar_chart(
                history.set_index("Filename")["Defect Count"]
            )

        st.subheader("Inspection Records")
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# 5. REPORTS
# ==================================================
elif page == "Reports":

    st.title("Inspection Reports")
    history = history_dataframe()

    if history.empty:
        st.info(
            "No reports available in this session. "
            "Run an inspection first."
        )
    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Inspection Report (CSV)",
            data=csv_data,
            file_name="defectiq_inspection_report.csv",
            mime="text/csv",
            type="primary",
        )

        if st.button("Save Report to Project Folder"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            report_path = (
                REPORTS_DIR / f"inspection_report_{timestamp}.csv"
            )

            history.to_csv(report_path, index=False)
            st.success(f"Report saved to: {report_path}")


# ==================================================
# 6. MODEL HEALTH
# ==================================================
elif page == "Model Health":

    st.title("Model Health")
    st.write(
        "Check the model file and software dependencies. "
        "File presence alone does not guarantee good predictions."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Python", sys.version.split()[0])
    c2.metric("OpenCV", cv2.__version__)
    c3.metric("Pandas", pd.__version__)

    model_path = MODELS_DIR / "best.pt"

    st.subheader("Trained Model")

    if model_path.exists():
        st.success("Model file found: models/best.pt")
        st.caption(
            f"File size: {model_path.stat().st_size / (1024**2):.2f} MB"
        )
    else:
        st.error(
            "models/best.pt was not found. "
            "Confirm the trained model file location."
        )

    st.subheader("Required Libraries")

    try:
        import ultralytics
        st.success(
            f"Ultralytics installed: {ultralytics.__version__}"
        )
    except ImportError:
        st.error(
            "Ultralytics is not installed in this Python environment."
        )

    st.success(f"NumPy installed: {np.__version__}")

    st.caption(
        "Use the saved test-set evaluation results for precision, "
        "recall, and mAP metrics."
    )
