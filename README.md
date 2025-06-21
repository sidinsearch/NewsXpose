# Combined Model Application

This project uses Docker to package all dependencies and make deployment easier, especially on platforms like Render.

## Quick Start

### Windows
```powershell
# Run the build and deploy script
.\build_and_deploy.ps1
```

### Linux/Mac
```bash
# Make the script executable
chmod +x build_and_deploy.sh

# Run the build and deploy script
./build_and_deploy.sh
```

## Manual Docker Commands

### Build and run locally
```bash
# Build the Docker image
docker build -t combined-model-app:latest .

# Run the container
docker run -p 8501:8501 -e PORT=8501 combined-model-app:latest
```

The application will be available at http://localhost:8501

### Using docker-compose
```bash
docker-compose up --build
```

## Deployment to Render

### Option 1: Deploy via Render Dashboard (Recommended)

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. In the Render dashboard, click "New +" and select "Web Service"
3. Connect your Git repository
4. Select "Docker" as the environment
5. Set the following:
   - Name: combined-model-app (or your preferred name)
   - Environment: Docker
   - Branch: main (or your default branch)
   - Plan: Free (or your preferred plan)
6. Click "Create Web Service"

### Option 2: Deploy via Render Blueprint

1. Push your code with the render.yaml file to a Git repository
2. In the Render dashboard, click "New +" and select "Blueprint"
3. Connect your Git repository
4. Render will automatically detect the render.yaml file and set up the service

## Why Docker?

Using Docker for this project provides several advantages:

1. **Consistent Environment**: The same environment is used for development and production
2. **Dependency Management**: All dependencies are packaged together, avoiding installation issues
3. **Faster Deployment**: Render can pull the pre-built image instead of building from scratch
4. **Resource Efficiency**: The multi-stage build creates a smaller final image

## Technical Details

### Dockerfile Structure

- **Multi-stage build**: Separates build dependencies from runtime dependencies
- **Virtual environment**: Uses Python venv for clean dependency management
- **Optimized layers**: Minimizes image size and improves build caching
- **Environment variables**: Uses PORT environment variable with fallback to 8501

### Requirements

All Python dependencies are specified in requirements.txt with pinned versions for reproducibility.