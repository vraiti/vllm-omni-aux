# OpenShift Pod Deployment

All resources must be deployed in the `vraiti` namespace.

## Prerequisites

- Logged in to the OpenShift cluster (`oc login`)
- Active project set to `vraiti` (`oc project vraiti`)

## Deploying

```bash
oc apply -f instructions/openshift/vllm-omni-pod.yaml -n vraiti
```

The pod uses `vllm/vllm-openai:v0.24.0` and runs `sleep infinity`.
Once running, `oc rsh -n vraiti <pod-name>` into it to clone repos, install
dependencies, and start the server.

## Polling for Readiness

```bash
bash instructions/openshift/poll-pod-server.sh <pod-name>
```

This checks both pod liveness (restart count, phase) and server
readiness (health endpoint) without requiring a port-forward.

## Cluster Details

- Server: `apps.alpha.modelarch.org`
- Namespace: `vraiti`
- HF cache: hostPath `/var/mnt/data/huggingface` mounted at `/hf-cache`
- The pod runs privileged to enable it to access the hostPath

Available GPU types and node assignments change over time. Before
deploying, inspect the cluster to determine which GPUs are available
and select a node with sufficient capacity for your workload:

```bash
oc get nodes -l nvidia.com/gpu.product -L nvidia.com/gpu.product
```

Update the `nodeSelector` in `vllm-omni-pod.yaml` and the
`CUDA_VISIBLE_DEVICES` / resource limits to match the chosen node.

## Teardown

```bash
oc delete pod <pod-name> -n vraiti
```
