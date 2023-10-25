import os
import pickle
import streamlit as st
import pandas as pd
from sklearn import set_config
import tempfile
import zipfile


def path_to_image_html(path):
    return '<img src="file:///' + path + '" width="60" >'


def convert_df(input_df, image_column):
    return input_df.to_html(index=False, escape=False, formatters={image_column: path_to_image_html})


set_config(display="html")

if "upload_file_id" not in st.session_state:
    st.session_state.upload_file_id = None
if "test_dataset" not in st.session_state:
    st.session_state.test_dataset = None
if "automl" not in st.session_state:
    st.session_state.automl = None


st.markdown(
    "<h1 style='text-align: center;'>🚀 Deploy ML Pipeline</h1>",
    unsafe_allow_html=True,
)


st.divider()

st.header("1. Upload a CSV or ZIP File", anchor=False)
st.info(
    """
    If you  upload a .zip file, it must contain a CSV file named "data.csv".
    It may also contain image files that are referenced in a column named "image_path" in the data.csv file.
    """,
    icon="ℹ️",
)
uploaded_dataset = st.file_uploader("Upload Your CSV File", type=["csv", "zip"])
if uploaded_dataset:
    print("Uploaded file:", uploaded_dataset)
    if st.session_state.upload_file_id != uploaded_dataset.file_id:
        print("Opening file:", uploaded_dataset.name, uploaded_dataset.type)
        if uploaded_dataset.type == "text/csv":
            test_dataset = pd.read_csv(uploaded_dataset)
        elif uploaded_dataset.type == "application/zip":
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_dataset) as zip_ref:
                zip_ref.extractall(tmp_dir)
            print("Extracted zip file to", tmp_dir)
            test_dataset = pd.read_csv(os.path.join(tmp_dir, "data.csv"))
            if "image_path" in test_dataset.columns:
                test_dataset["image_path"] = test_dataset["image_path"].apply(
                    lambda x: os.path.join(tmp_dir, x)
                )
        else:
            st.write("Unsupported file type:", uploaded_dataset.type)
            st.stop()

        st.session_state.upload_file_id = uploaded_dataset.file_id
        st.session_state.test_dataset = test_dataset
    else:
        test_dataset = st.session_state.test_dataset

    st.dataframe(test_dataset, hide_index=True)

    st.header("2. Upload a ML Pipeline", anchor=False)
    uploaded_pipeline = st.file_uploader("Upload Your ML Pipeline", type=["pkl"])

    y_pred = None
    if uploaded_pipeline:
        pipeline = pickle.load(uploaded_pipeline)
        st.header("3. Make Predictions", anchor=False)
        st.markdown("<p>Run your ML pipeline to make predictions.</p>", unsafe_allow_html=True)
        if st.button("Run!"):
            with st.spinner("Making predictions..."):
                y_pred = pipeline.predict(test_dataset)
                print("Predictions", y_pred)

    if y_pred is not None:
        predictions = pd.DataFrame({'predictions': y_pred})
        output = pd.concat([test_dataset, predictions], axis=1)

        #st.dataframe(output, hide_index=True)
        output['image'] = output['image_path']
        html = convert_df(output, 'image')
        st.write(html, unsafe_allow_html=True)

st.divider()

