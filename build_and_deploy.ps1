# PowerShell script for building and deploying the Docker image

# Check for model compatibility issues and fix them
Write-Host "Checking for model compatibility issues..." -ForegroundColor Green
python update_model_loading.py

# Test model loading
Write-Host "Testing model loading..." -ForegroundColor Green
python test_model_loading.py

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t combined-model-app:latest .

# Test the Docker image locally
Write-Host "Running Docker container locally..." -ForegroundColor Green
Write-Host "The application will be available at http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the container" -ForegroundColor Yellow
docker run -p 8501:8501 -e PORT=8501 combined-model-app:latest

# Instructions for deploying to Render
Write-Host "`nTo deploy to Render:" -ForegroundColor Green
Write-Host "1. Push your code to a Git repository" -ForegroundColor Cyan
Write-Host "2. In the Render dashboard, click 'New +' and select 'Web Service'" -ForegroundColor Cyan
Write-Host "3. Connect your Git repository" -ForegroundColor Cyan
Write-Host "4. Select 'Docker' as the environment" -ForegroundColor Cyan
Write-Host "5. Set the name and plan" -ForegroundColor Cyan
Write-Host "6. Click 'Create Web Service'" -ForegroundColor Cyan