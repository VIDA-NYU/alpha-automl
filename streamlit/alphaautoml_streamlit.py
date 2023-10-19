import pickle
import streamlit as st
import pandas as pd
from alpha_automl import AutoMLClassifier
from sklearn import set_config
from sklearn.utils import estimator_html_repr
set_config(display='html')


st.markdown("<h1 style='text-align: center; '>Alpha-AutoML</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; '> An extensible open-source AutoML system that supports multiple ML tasks. </p>", unsafe_allow_html=True)
st.divider()

st.header("Upload Your CSV File", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])
train_dataset = None
automl = None

if uploaded_file:
    train_dataset = pd.read_csv(uploaded_file)
    st.dataframe(train_dataset, hide_index=True)

st.header("Search ML Pipelines", anchor=False)

if uploaded_file:
    target_column = st.text_input('Column to predict?')
    time_bound = st.slider('How long run the search?', 1, 30, 10)
    #target_column = 'target'
    #time_bound = 5
    print('>>>, time_bound', time_bound)
    if target_column and time_bound:
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]

    if st.button('Search'):
        print("Initializing AutoML...")
        automl = AutoMLClassifier('./tmp', time_bound=time_bound, start_mode="spawn", verbose=True)
        print("Searching models...")
        automl.fit(X_train, y_train)
        print("Done.")
        if len(automl.pipelines) > 0:
            st.write('Pipelines Leaderboard:')
            st.dataframe(automl.get_leaderboard(), hide_index=True)
        else:
            st.write("No valid pipelines found.")


st.header("Export ML model", anchor=False)

if automl is not None and len(automl.pipelines) > 0:
    st.write('Pipeline Structure:')
    st.components.v1.html(estimator_html_repr(automl.get_pipeline()))
    st.write('Save the model!')
    st.download_button(
        "Download Now",
        data=pickle.dumps(automl.get_pipeline()),
        file_name="best_model.pkl",
    )

st.divider()