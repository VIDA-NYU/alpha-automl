FROM python:3.11.6 AS alpha-automl-jupyterhub

# Install JupyterHub and dependencies
RUN pip3 --disable-pip-version-check install --no-cache-dir \
    jupyterhub==4.0.2 \
    notebook==7.0.6 \
    jupyterlab==4.0.8 \
    jupyterlab-server==2.25.0

# Install AlphaD3M and dependencies
ADD . /alpha-automl/
WORKDIR /alpha-automl/
ARG BUILD_OPTION
RUN if [ -n "$BUILD_OPTION" ]; then \
      pip3 install --no-cache-dir -e .[$BUILD_OPTION]; \
    else \
      pip3 install --no-cache-dir -e .; \
    fi

# Create a user, since we don't want to run as root
RUN useradd -m alphaautoml
ENV HOME=/home/alphaautoml
WORKDIR $HOME
USER alphaautoml
COPY --chown=alphaautoml examples /home/alphaautoml/examples

# Huggingface text config 
ENV TOKENIZERS_PARALLELISM=false

FROM alpha-automl-jupyterhub AS alpha-automl
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--no-browser"]
