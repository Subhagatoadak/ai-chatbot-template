# Stage 1: Build the React app
FROM node:16-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package.json ./
RUN npm install

# Copy the rest of the application and build it
COPY . ./
RUN npm run build

# Install nodemon globally for auto-restart
RUN npm install -g nodemon

# Stage 2: Serve the built app using Nginx
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]