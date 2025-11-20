# Multi-stage build for production-optimized frontend

# Stage 1: Build
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY . .

# Build arguments for environment variables
# Hospital A
ARG VITE_MANAGE_API_URL_HOSPITAL_A
ARG VITE_SCRIBE_API_URL_HOSPITAL_A
ARG VITE_SUMMARIZER_API_URL_HOSPITAL_A
# Hospital B
ARG VITE_MANAGE_API_URL_HOSPITAL_B
ARG VITE_SCRIBE_API_URL_HOSPITAL_B
ARG VITE_SUMMARIZER_API_URL_HOSPITAL_B
# Shared services
ARG VITE_DOL_API_URL
ARG VITE_FEDERATION_API_URL
ARG VITE_ENVIRONMENT=production

# Set environment variables for build
ENV VITE_MANAGE_API_URL_HOSPITAL_A=${VITE_MANAGE_API_URL_HOSPITAL_A}
ENV VITE_SCRIBE_API_URL_HOSPITAL_A=${VITE_SCRIBE_API_URL_HOSPITAL_A}
ENV VITE_SUMMARIZER_API_URL_HOSPITAL_A=${VITE_SUMMARIZER_API_URL_HOSPITAL_A}
ENV VITE_MANAGE_API_URL_HOSPITAL_B=${VITE_MANAGE_API_URL_HOSPITAL_B}
ENV VITE_SCRIBE_API_URL_HOSPITAL_B=${VITE_SCRIBE_API_URL_HOSPITAL_B}
ENV VITE_SUMMARIZER_API_URL_HOSPITAL_B=${VITE_SUMMARIZER_API_URL_HOSPITAL_B}
ENV VITE_DOL_API_URL=${VITE_DOL_API_URL}
ENV VITE_FEDERATION_API_URL=${VITE_FEDERATION_API_URL}
ENV VITE_ENVIRONMENT=${VITE_ENVIRONMENT}

# Build the application
RUN npm run build

# Stage 2: Production server
FROM nginx:alpine AS production

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:8080/ || exit 1

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

