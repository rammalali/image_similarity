# Frontend Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy frontend files
COPY . .

# Expose port 8080
EXPOSE 8080

# Run http-server
CMD ["npm", "run", "serve"]

