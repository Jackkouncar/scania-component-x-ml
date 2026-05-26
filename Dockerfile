FROM node:22-bookworm-slim

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5173
ENV APP_DB_FILE=/app/database/eneb453_app.sqlite

COPY package*.json ./
RUN npm ci --omit=dev

COPY . .
RUN npm run build:frontend

EXPOSE 5173

CMD ["npm", "run", "serve"]
