# Alpha-AutoML Streamlit App

This directory contains a Streamlit app that implements model training and prediction using the `alpha-automl` system.

## Required dependencies

Make sure to install the following dependencies:
- alpha-automl[image]
- streamlit

We recommend creating a virtual environment to avoid dependency conflicts:
```
python -m venv venv
source venv/bin/activate
pip install alpha-automl[image]@"git+https://github.com/VIDA-NYU/alpha-automl@devel"
pip install streamlit
```

In the commands above, we are installing a specific version of alpha-automl, including the optional `image` dependencies that are required to support building models for image classification.

## Running the app

To run the app execute the following command:

```
streamlit run Home.py
```

The command will print a message like this:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.150:8501
```
Now you can open the app using your browser using one of the addresses specified above, e.g., http://localhost:8501.