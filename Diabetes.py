import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn import datasets

# Page Config
st.set_page_config(
    page_title="Diabetes Data Exploration",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

# Convert sex to readable category
df["sex_cat"] = df["sex"].apply(lambda v: "male" if v > 0 else "female")

# Feature name
feature_name_map = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "Body Mass Index (BMI)",
    "bp": "Blood Pressure",
    "s1": "Total Serum Cholesterol (TC)",
    "s2": "Low-Density Lipoprotein (LDL)",
    "s3": "High-Density Lipoprotein (HDL)",
    "s4": "Triglycerides (TG)",
    "s5": "Serum Glucose",
    "s6": "Serum TSH"
}

readable_to_original = {v: k for k, v in feature_name_map.items()}
readable_features = list(feature_name_map.values())

# Theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 60%);
        color: #1b3250;
    }
    section[data-testid="stSidebar"] {
        background-color: #eef6ff;
        border-right: 1px solid #d0e3f7;
    }
    h1, h2, h3 {
        color: #0f2b4a;
    }
    .plot-card {
        background-color: rgba(255,255,255,0.85);
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(22,40,80,0.06);
        margin-bottom: 12px;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
        height: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Data Selection")

# Feature selectors with readable names
x_feature_readable = st.sidebar.selectbox(
    "X-Axis Feature",
    options=readable_features,
    index=2  # default BMI
)

y_feature_readable = st.sidebar.selectbox(
    "Y-Axis Feature",
    options=readable_features,
    index=3  # default BP
)

x_feature = readable_to_original[x_feature_readable]
y_feature = readable_to_original[y_feature_readable]

# Target filter
min_target = int(df["target"].min())
max_target = int(df["target"].max())
target_range = st.sidebar.slider(
    "Target Range (Disease Progression)",
    min_value=min_target,
    max_value=max_target,
    value=(min_target, max_target)
)

# Colour by sex
colour_by_sex = st.sidebar.checkbox("Colour by Gender", value=True)

# Toggle raw data
show_raw = st.sidebar.checkbox("Show raw filtered data", value=False)

# Apply filter
filtered = df[
    (df["target"] >= target_range[0]) &
    (df["target"] <= target_range[1])
].copy()

# Page header
st.title("Diabetes Dataset Exploration Dashboard")
st.write(
    "Interactively explore relationships between clinical features and disease progression using scatter plots, histograms, and statistical summaries."
)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Filtered Rows", len(filtered))
m2.metric("Mean Target", f"{filtered['target'].mean():.2f}")
m3.metric("Mean BMI", f"{filtered['bmi'].mean():.2f}")

st.markdown("---")

# Satter Plot
with st.expander("Scatter Plot of Selected Features", expanded=True):
    st.write(
        "This scatter plot shows interactions between two clinical measures. Enable gender colouring for clearer subgroup patterns."
    )

    fig_scatter = px.scatter(
        filtered,
        x=x_feature,
        y=y_feature,
        color="sex_cat" if colour_by_sex else None,
        labels={
            x_feature: feature_name_map[x_feature],
            y_feature: feature_name_map[y_feature],
            "sex_cat": "Gender"
        },
        color_discrete_map={"male": "#1958a3", "female": "#69a6d8"},
        hover_data=["target", "sex_cat"],
        title=f"{feature_name_map[x_feature]} vs {feature_name_map[y_feature]}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")
    st.markdown('</div>', unsafe_allow_html=True)

# Histogram
with st.expander("Histogram of Disease Progression", expanded=False):
    st.write(
        "This histogram displays the distribution of diabetes progression levels. "
        "This helps identify skewness, peaks or potential outliers."
    )

    fig_hist = px.histogram(
        filtered,
        x="target",
        nbins=30,
        title="Distribution of Disease Progression",
        labels={"target": "Disease Progression Score"}
    )

    fig_hist.update_traces(opacity=0.75)
    st.plotly_chart(fig_hist, use_container_width=True, key="hist")
    st.markdown('</div>', unsafe_allow_html=True)

# Pairwise
with st.expander("Pairwise Snapshot (BMI, BP, Target)", expanded=False):
    st.write("A quick multi-feature matrix for three key indicators: BMI, BP and target.")

    pair_feats = ["bmi", "bp", "target"]
    small_df = filtered[pair_feats].rename(columns=feature_name_map)

    fig_pair = px.scatter_matrix(
        small_df,
        dimensions=list(small_df.columns),
        title="Pairwise Feature Overview"
    )
    st.plotly_chart(fig_pair, use_container_width=True, key="pair")
    st.markdown('</div>', unsafe_allow_html=True)

# Data Sumamry
with st.expander("Data Summary", expanded=False):
    st.subheader("Descriptive Statistics")
    st.write("Statistical overview of the filtered subset.")
    st.dataframe(filtered.describe().round(3))

    st.markdown("### Counts by Gender")
    st.table(filtered["sex_cat"].value_counts().rename_axis("Gender").reset_index(name="Count"))

    if show_raw:
        st.markdown("### Raw Data of the First 200 Rows")
        st.dataframe(filtered.reset_index(drop=True).head(200))

# Footer
st.markdown("---")
st.caption("Developed by Daniel Muhammad Timms | Bachelor of Computer Science")
