## HELM Jupyterhub Deployment
- Get access to HSRN namespace via this tutorial: https://k8s-docs.hsrn.nyu.edu/get-started/
- Go to kubernetes directory
- Create ingress server:
  `kubectl create -f ingress.yml`
- Create pod using HELM upgrade command 
  `helm upgrade --cleanup-on-fail --install alpha-automl jupyterhub/jupyterhub --namespace YOUR_NAMESPACE --values values.yaml`
- Check if the pod is successfully created using:
  `kubectl get pod -n alphad3m`

## Docker Build
- We can use a build option (full, timeseries, nlp...) upon docker build for different building plans:
  `docker build --build-arg BUILD_OPTION=full .`
- Or simply a base version using:
  `docker build .`

## Docker Run
- To let Jupyter auto-generate a token:
  `docker run -p 8888:8888 ghcr.io/vida-nyu/alpha-automl`
- To run using a custom security token:
  `docker run -p 8888:8888 -e JUPYTER_TOKEN="<my-token>" ghcr.io/vida-nyu/alpha-automl`
- If the service is running in a secure environment, the authentication can be disabled:
  `docker run -p 8888:8888 ghcr.io/vida-nyu/alpha-automl --NotebookApp.token=''`