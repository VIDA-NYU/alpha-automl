import streamlit as st
from sklearn import set_config


set_config(display="html")

st.set_page_config(
    page_title="Alpha-AutoML App",
)


with st.columns(3)[1]:
    st.image("https://github.com/VIDA-NYU/alpha-automl/raw/devel/Alpha-AutoML_logo.png")

st.markdown(
    "<p style='text-align: center;'>An extensible open-source AutoML system that supports multiple ML tasks </p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown(
    "<p style='text-align: justify;'>Alpha-AutoML leverages in reinforcement learning and neural network components "
    "and it relies on standard, open-source infrastructure to specify and run pipelines. It is compatible with "
    "state-of-the-art ML techniques: by using the Sklearn pipeline infrastructure, Alpha-AutoML is fully compatible "
    "with other standard libraries like XGBoost, Hugging Face, Keras, PyTorch. In addition, primitives can be added on "
    "the fly through the standard Sklearn’s fit/predict API, making it possible for Alpha-AutoML to leverage new "
    "developments in ML and keep up with the fast pace in the area. </p>",
    unsafe_allow_html=True,
)

st.divider()
