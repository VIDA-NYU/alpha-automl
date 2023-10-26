import os
import pickle
import streamlit as st
import pandas as pd
from sklearn import set_config
import tempfile
import zipfile
from PIL import Image
import base64
from io import BytesIO


def path_to_image_html(path):
    # load image and create thumbnail
    image = Image.open(path)
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    # convert image to base64
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # embbed base64 image data in the html tag
    return f'<img src="data:image/jpeg;base64,{img_str}" >'


def convert_df(input_df, image_column):
    return input_df.to_html(
        index=False, escape=False, formatters={image_column: path_to_image_html}
    )


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

st.header("1. Upload data for predictions", anchor=False)
st.info(
    """
    You can upload a CSV or ZIP file for which you want to make predictions. If you select
    a ZIP file, make sure it has a CSV file named "data.csv" that the model will use to 
    make predictions. The CSV file should have the same columns as the training data 
    used to create the model, except for the prediction target column.
    """,
    icon="ℹ️",
)
uploaded_dataset = st.file_uploader("Upload Your CSV/ZIP File", type=["csv", "zip"])
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
        st.markdown(
            "<p>Run your ML pipeline to make predictions.</p>", unsafe_allow_html=True
        )
        if st.button("Run!"):
            with st.spinner("Making predictions..."):
                y_pred = pipeline.predict(test_dataset)
                print("Predictions", y_pred)

    if y_pred is not None:
        predictions = pd.DataFrame({"predictions": y_pred})
        output = pd.concat([test_dataset, predictions], axis=1)

        if "image_path" not in output.columns:
            st.dataframe(output, hide_index=True)
        else:
            output["image_path (preview)"] = output["image_path"]
            html = convert_df(output, "image_path (preview)")
            st.write(html, unsafe_allow_html=True)
st.divider()
