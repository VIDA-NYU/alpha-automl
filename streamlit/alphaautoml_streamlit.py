import pickle
import streamlit as st
import pandas as pd
from alpha_automl import AutoMLClassifier
from sklearn import set_config
from sklearn.utils import estimator_html_repr

set_config(display="html")

if "train_dataset" not in st.session_state:
    st.session_state.train_dataset = None

if "automl" not in st.session_state:
    st.session_state.automl = None

with st.columns(3)[1]:
    st.image("https://github.com/VIDA-NYU/alpha-automl/raw/devel/Alpha-AutoML_logo.png")

st.markdown(
    "<p style='text-align: center;'>An extensible open-source AutoML system that supports multiple ML tasks </p>",
    unsafe_allow_html=True,
)

st.divider()

st.header("1. Upload Your CSV File", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])
if uploaded_file:
    if st.session_state.train_dataset is None:
        train_dataset = pd.read_csv(uploaded_file)
    else:
        train_dataset = st.session_state.train_dataset

    st.dataframe(train_dataset, hide_index=True)

    st.header("2. Search ML Pipelines", anchor=False)

    headers = tuple(train_dataset.columns)
    target_column = st.selectbox(
        "Column to predict?",
        headers,
        index=None,
        placeholder="Select the target column...",
    )
    time_bound = st.slider("How long run the search (minutes)?", 1, 30, 5)

    if target_column and time_bound:
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]

    if st.button("Search"):
        print("Initializing AutoML...")
        automl = AutoMLClassifier(
            "./tmp", time_bound=time_bound, start_mode="spawn", verbose=True
        )
        print("Searching models...")
        automl.fit(X_train, y_train)
        print("Done.")
        if len(automl.pipelines) > 0:
            st.session_state.automl = automl
        else:
            st.write("No valid pipelines found.")

    automl = st.session_state.automl

    if automl:
        st.write("Pipelines leaderboard:")
        st.dataframe(automl.get_leaderboard(), hide_index=True)

    if automl and len(automl.pipelines) > 0:
        st.header("3. Select a pipeline", anchor=False)
        pipeline_id = st.selectbox(
            "Select a pipeline:",
            tuple(automl.pipelines.keys()),
            index=0,
            placeholder="Select one pipeline to proceed...",
        )
        print(f"Selected pipeline: {pipeline_id}")
        pipeline = automl.get_pipeline(pipeline_id)

        st.write("Pipeline structure:")
        st.components.v1.html(estimator_html_repr(pipeline))

        st.header("4. Export ML model", anchor=False)
        st.write(
            "You can now save the model file (model.pkl) and re-load it in Python code using the snippet below!"
        )
        st.download_button(
            "Download!",
            data=pickle.dumps(pipeline),
            file_name="model.pkl",
        )
        st.write(
            """
        ```python
        ### FIXME fix this script
        import pickle
        
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        X = ... # data
        model.predict(X)
        ```
        """
        )

st.divider()
