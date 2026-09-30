
docker compose down
docker compose up -d
docker compose ps		
docker compose exec postgres psql -U agent -d agents -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker compose exec postgres psql -U agent -d agents -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"