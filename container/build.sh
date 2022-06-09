#!/bin/bash

SVC='byoc-tensorflow'
REGION='eu-west-1'
ACCOUNT_ID=$(aws sts get-caller-identity --output text --query 'Account')
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

## Check if exists in ECR
check=$(aws ecr describe-repositories --region ${REGION} --repository-name ${SVC} | jq -r .repositories[].repositoryName | wc -l)
if [[ $check -eq 0 ]]; then
  aws ecr --region ${REGION} create-repository --repository-name ${SVC}
fi

# Build
docker build --no-cache -t $ECR_URL/$SVC:v2.5.0 .

# Login
$(aws ecr get-login --no-include-email --region $REGION)

# Push
docker push $ECR_URL/$SVC:latest