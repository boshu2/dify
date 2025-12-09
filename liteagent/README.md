# LiteAgent

A lightweight LLM agent platform built from scratch. Supports multiple LLM providers, data sources, and a simple agent builder with chat interface.

## Features

- **LLM Marketplace**: Connect to OpenAI, Anthropic, or Ollama
- **Data Sources**: Add files, URLs, or plain text as knowledge bases
- **Agent Builder**: Create agents with dropdowns - pick provider, attach data, set prompts
- **Chat Interface**: Test your agents with a built-in chat UI

## Quick Start

### Backend

```bash
cd backend

# Install dependencies
pip install -e .

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install
# or
pnpm install

# Run the dev server
npm run dev
# or
pnpm dev
```

Then open http://localhost:3000

## Architecture

```
liteagent/
├── backend/
│   └── app/
│       ├── api/routes/      # REST API endpoints
│       ├── core/            # Config, database
│       ├── models/          # SQLAlchemy models
│       ├── providers/       # LLM and DataSource providers
│       │   ├── llm/         # OpenAI, Anthropic, Ollama
│       │   └── datasource/  # File, URL, Text
│       ├── schemas/         # Pydantic schemas
│       └── services/        # Business logic
└── frontend/
    └── src/
        ├── app/             # Next.js pages
        ├── components/      # React components
        ├── lib/             # API client, utilities
        └── types/           # TypeScript types
```

## API Endpoints

- `GET/POST /api/providers/` - LLM provider management
- `GET/POST /api/datasources/` - Data source management
- `GET/POST /api/agents/` - Agent CRUD
- `POST /api/agents/{id}/chat` - Chat with an agent
- `GET /api/meta/provider-types` - Available provider types
- `GET /api/meta/datasource-types` - Available data source types

## Configuration

Create a `.env` file in the backend directory:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
```

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite, Pydantic
- **Frontend**: Next.js 14, React 18, TanStack Query, Tailwind CSS
- **LLM SDKs**: openai, anthropic, httpx (for Ollama)

## License

MIT
