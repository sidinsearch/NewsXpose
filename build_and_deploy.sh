#!/bin/bash

# Check for model compatibility issues and fix them
echo -e "\e[32mChecking for model compatibility issues...\e[0m"
python update_model_loading.py

# Test model loading
echo -e "\e[32mTesting model loading...\e[0m"
python test_model_loading.py

# Build the Docker image
echo -e "\e[32mBuilding Docker image...\e[0m"
docker build -t combined-model-app:latest .

# Test the Docker image locally
echo -e "\e[32mRunning Docker container locally...\e[0m"
echo -e "\e[33mThe application will be available at http://localhost:8501\e[0m"
echo -e "\e[33mPress Ctrl+C to stop the container\e[0m"
docker run -p 8501:8501 -e PORT=8501 combined-model-app:latest

# Instructions for deploying to Render
echo -e "\n\e[32mTo deploy to Render:\e[0m"
echo -e "\e[36m1. Push your code to a Git repository\e[0m"
echo -e "\e[36m2. In the Render dashboard, click 'New +' and select 'Web Service'\e[0m"
echo -e "\e[36m3. Connect your Git repository\e[0m"
echo -e "\e[36m4. Select 'Docker' as the environment\e[0m"
echo -e "\e[36m5. Set the name and plan\e[0m"
echo -e "\e[36m6. Click 'Create Web Service'\e[0m"