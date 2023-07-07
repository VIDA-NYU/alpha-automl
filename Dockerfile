FROM python:3.10.11

# Install JupyterHub and dependencies
RUN pip3 --disable-pip-version-check install --no-cache-dir \
    jupyterhub==3.1.1 \
    notebook==6.5.2 \
    jupyterlab==3.5.3 \
    jupyterlab-server==2.16.0

# Install AlphaD3M and dependencies
# RUN pip3 --disable-pip-version-check install --no-cache-dir \
# 	alpha-automl

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

ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--no-browser"]