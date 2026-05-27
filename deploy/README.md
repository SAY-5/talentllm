# Deployment notes

The intended target is AWS. This directory documents how the service is packaged
and the resources it expects. It does not perform a live deployment and contains
no provisioned infrastructure.

## Container image

The `Dockerfile` builds a slim image that installs the package and serves the
FastAPI app with uvicorn on port 8000. The default provider is offline, so the
image needs no model credentials to run.

```bash
docker build -t talentllm:local -f deploy/Dockerfile .
docker run -p 8000:8000 talentllm:local
```

## AWS layout

A typical deployment on AWS uses:

- Amazon ECR to store the container image.
- AWS Fargate (ECS) or App Runner to run the container as a service.
- An Application Load Balancer in front of the service for the `/ask` and
  `/health` routes.

`task-definition.json` is a sample ECS task definition for the image. Replace the
account id, region, and image tag placeholders before use. The health check path
is `/health`.

## Substituting a hosted provider

To use a hosted model provider instead of the offline default, implement the
`Provider` interface in `talentllm.provider` and pass an instance to `Assistant`.
Provider credentials should be supplied through environment variables or a secret
store such as AWS Secrets Manager rather than committed to the repository.
