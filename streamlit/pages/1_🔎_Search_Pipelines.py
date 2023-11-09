import os
import pickle
import streamlit as st
import pandas as pd
from alpha_automl import AutoMLClassifier
from sklearn import set_config
from sklearn.utils import estimator_html_repr
import tempfile
import zipfile


set_config(display="html")

if "upload_file_id" not in st.session_state:
    st.session_state.upload_file_id = None
if "train_dataset" not in st.session_state:
    st.session_state.train_dataset = None
if "automl" not in st.session_state:
    st.session_state.automl = None


st.markdown(
    "<h1 style='text-align: center;'>🔎 Search ML Pipelines </h1>",
    unsafe_allow_html=True,
)


st.divider()

st.header("1. Upload the Training Data", anchor=False)
st.info(
    """
    You may upload either a CSV or a ZIP file. If you upload a ZIP file, it
    must contain a CSV file named "data.csv" that contains the training data.
    If the 'data.csv' includes a column named "image_path" containing file 
    paths to images that are also included in the CSV file, the system will 
    automatically load process these images as well.
    """,
    icon="ℹ️",
)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv", "zip"])
if uploaded_file:
    print("Uploaded file:", uploaded_file)
    if st.session_state.upload_file_id != uploaded_file.file_id:
        print("Opening file:", uploaded_file.name, uploaded_file.type)
        if uploaded_file.type == "text/csv":
            train_dataset = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/zip":
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                zip_ref.extractall(tmp_dir)
            print("Extracted zip file to", tmp_dir)
            train_dataset = pd.read_csv(os.path.join(tmp_dir, "data.csv"))
            if "image_path" in train_dataset.columns:
                train_dataset["image_path"] = train_dataset["image_path"].apply(
                    lambda x: os.path.join(tmp_dir, x)
                )
        else:
            st.write("Unsupported file type:", uploaded_file.type)
            st.stop()

        st.session_state.upload_file_id = uploaded_file.file_id
        st.session_state.train_dataset = train_dataset
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
        if not target_column:
            st.error("Please select a target column!")
            st.stop()

        print("Initializing AutoML...")
        automl = AutoMLClassifier(time_bound=time_bound, start_mode="spawn")
        print("Searching models...")
        with st.spinner("Searching models..."):
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
            data=pickle.dumps(automl.get_serialized_pipeline(pipeline_id)),
            file_name="model.pkl",
        )
        st.write(
            """
        ```python
        #
        # 1. Make sure you have installed the required 'alpha-automl' package
        # 2. Run the following code to load the model and apply it to new data
        #
        import pickle
        
        model_file = "model.pkl" # the path to model file you downloaded
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        
        # Load the input data as a Pandas dataframe. It must have the same columns
        # as the training data used to build the model (except the target column)
        X = #...

        # The predict() function returns a numpy array with the predictions
        model.predict(X)
        ```
        """
        )

st.divider()
