# AI Agent Node

A TypeScript-based AI agent application with MongoDB, Express, and WebSocket support.

## Overview

This project provides an intelligent agent service that can be integrated with applications needing AI capabilities. It uses WebSocket connections for real-time communication and offers a RESTful API for management operations.

## Features

- Real-time AI agent communication via WebSockets
- RESTful API for configuration and management
- MongoDB integration for data persistence
- Docker support for development and production environments
- Comprehensive logging system
- Authentication and security features

## Technology Stack

- **Backend**: Node.js, Express, TypeScript
- **Database**: MongoDB with Mongoose ODM
- **AI Integrations**: LangChain, OpenAI, Google Generative AI
- **Real-time Communication**: WebSockets (ws)
- **Containerization**: Docker, Docker Compose
- **Documentation**: Swagger/OpenAPI
- **Testing**: Jest

## Installation and Setup

### Prerequisites

- Node.js (v16+)
- npm or pnpm
- MongoDB (local or remote)
- OpenAI API key (for AI functionality)

### Local Development Setup

1. Clone the repository:

````
git clone https://github.com/your-username/AiAgentNode.git
cd AiAgentNode

## Access Docker Exposed Ports and Related Information

When running Docker containers, it's important to understand how to access services running on exposed ports. Here are some key points:

1. **Accessing Services**: You can access services running in Docker containers using the container's IP address and the exposed port. For example, if a container is running on port 3000, you can access it using the IP address of the Docker host and the port number.

2. **Port Mapping**: Docker uses a port mapping system to allow external access to services running inside containers. For example, if you map a port in Docker (e.g., `-p 8080:3000`), you can access the service on your host machine's port 8080.

3. **Common Scenarios**:
   - **Development**: When running the application in Docker for development, you might map the container's port to a port on your host machine for easier access.
   - **Production**: In a production environment, you might expose certain services to the outside world for monitoring or external access.

4. **Troubleshooting**: If you encounter issues accessing a service, check the Docker logs for any port-related errors. You might need to adjust the port mapping or check firewall settings.

## Accessing Docker Exposed Ports

To access a service running in a Docker container, follow these steps:

1. **Find the Container ID**: Use the `docker ps` command to find the container ID of the running service.

# View running containers
docker ps

# View logs for a container
docker logs <container_id>

# Execute a command in a running container
docker exec -it <container_id> /bin/bash

# Stop all running containers
docker-compose down

# Rebuild containers after code changes
docker-compose up -d --build

# View Docker container resource usage
docker stats

# Remove unused Docker resources (cleanup)
docker system prune -a

# View Docker images
docker images

# Remove a specific Docker image
docker rmi <image_id>

# Check Docker Compose configuration
docker-compose config

# Restart a specific service
docker-compose restart <service_name>

# Scale a specific service
docker-compose up -d --scale <service_name>=<num_instances>

# Build and start the development containers
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Build production image
docker build -t ai-agent-node:prod -f Dockerfile.prod .

# Run container from production image
docker run -p 3000:3000 ai-agent-node:prod

## Finding and Using Docker Container IP Addresses

When working with Docker containers, sometimes you need to connect directly to a container using its IP address. Here's how to find and use container IP addresses:

### Method 1: Using `docker inspect`

The most reliable way to get a container's IP address:

```bash
# Get detailed information about a container, including its IP address
docker inspect <container_id_or_name> | grep IPAddress

# For a more specific output
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id_or_name>
````

### Method 2: Using `docker network inspect`

If you want to see all containers in a specific network:

```bash
# First, list all networks
docker network ls

# Then inspect a specific network
docker network inspect <network_name>
```

### Method 3: From within a container

You can also get the IP address from inside the container:

```bash
# Start a shell in the container
docker exec -it <container_id_or_name> /bin/bash

# Then inside the container run
hostname -i
# or
ip addr show
```

### Container IP Address Use Cases

1. **Container-to-Container Communication**:

   - Within the same Docker network, containers can communicate using their IP addresses
   - Better practice: Use Docker's built-in DNS service by referencing containers by name

2. **Debugging Network Issues**:

   - When troubleshooting connectivity problems between containers
   - Testing if a service is properly bound to its network interface

3. **Custom Network Configurations**:
   - When implementing custom overlay networks
   - For specialized networking requirements

### Important Notes

- Container IP addresses are **dynamic** and may change when containers restart
- For stable connections between services, use Docker's networking features:

  - Docker Compose service names
  - Docker Swarm service discovery
  - User-defined bridge networks

- In most scenarios, you should use the service name for container-to-container communication:

  ```
  # Example with MongoDB container named "mongodb"
  MONGODB_URI=mongodb://mongodb:27017/ai-agent
  ```

- If you're using Docker Compose, containers in the same compose file can communicate using the service name as hostname
