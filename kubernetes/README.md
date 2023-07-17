## HELM Jupyterhub Deployment
- If using NYU HSRN server, get access to HSRN namespace via this tutorial: https://k8s-docs.hsrn.nyu.edu/get-started/
- Go to kubernetes directory
- If you are using NYU HSRN, create ingress server, else **please use the ingress config of your server**:
  `kubectl create -f ingress.yml`
- Let HELM command line tool know about a HELM chart repository that we decide to name jupyterhub.
  ```
  helm repo add jupyterhub https://hub.jupyter.org/helm-chart/
  helm repo update
  ```
- Create pod using HELM upgrade command 
  `helm upgrade --cleanup-on-fail --install alpha-automl jupyterhub/jupyterhub --namespace YOUR_NAMESPACE --values values.yaml`
- Check if the pod is successfully created using:
  `kubectl get pod -n alphad3m`
