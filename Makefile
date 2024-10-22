# Define Variables
APP_NAME = multimodal-chatbot
DOCKER_IMAGE = $(APP_NAME):latest
DOCKER_CONTAINER = $(APP_NAME)-container
PYTHON = python3

# Default target
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Build docker image
build:
	docker build -t $(DOCKER_IMAGE) .

# Deploy docker image with GPU access
deploy:
	docker run -it --rm --gpus all -v $(PWD):/workspace -w /workspace --name $(DOCKER_CONTAINER) $(DOCKER_IMAGE)

# Run python script
run:
	$(PYTHON) main.py

# Clean up temporary files and directories
.PHONY: clean
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
