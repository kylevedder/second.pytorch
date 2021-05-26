#!/bin/bash
docker build . -t second-pytorch-sparse && \
docker tag second-pytorch-sparse docker-registry.grasp.cluster/second-pytorch-sparse && \
docker push docker-registry.grasp.cluster/second-pytorch-sparse