#!/bin/bash

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login

# Get the image ID
IMAGE_ID=$(docker images -q 22521301_lab02_api)

# Tag the image
echo "Tagging image..."
docker tag $IMAGE_ID tanmaivan/housing-price-api:latest

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push tanmaivan/housing-price-api:latest

echo "Done!"