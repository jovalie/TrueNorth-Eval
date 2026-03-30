-- Bootstrap the second database used for chat history logs.
-- The primary chat-state database is created by POSTGRES_DB during container init.

SELECT 'CREATE DATABASE truenorth_logs'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'truenorth_logs')\gexec

GRANT ALL PRIVILEGES ON DATABASE truenorth_chat TO truenorth;
GRANT ALL PRIVILEGES ON DATABASE truenorth_logs TO truenorth;
