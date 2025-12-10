-- LiteAgent Database Initialization
-- This script runs on first database creation

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for common queries (will be managed by Alembic migrations)
-- This file is for initial setup only

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE liteagent TO postgres;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'LiteAgent database initialized successfully';
END $$;
