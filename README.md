<div align="center">
  <img src="./public/Logo.svg" alt="CCAdvance Logo" width="120" height="120">
  
  # CCAdvance
  
  **Advanced Web Crawling & RAG for Claude Code**
  
  *Intelligent documentation indexing and retrieval for AI-powered development*
</div>

---

## Overview

CCAdvance is a Model Context Protocol (MCP) server designed specifically for Claude Code users who need advanced web crawling and RAG capabilities. It enables Claude Code to intelligently crawl documentation sites, extract code examples, and provide contextual search across indexed content.

**Perfect for**: Indexing documentation sites, API references, and technical content for use within Claude Code development workflows.

## Claude Code Integration

### Quick Setup

1. **Start the server** (see [Installation](#installation) below)
2. **Add to Claude Code** using one of these options:

#### Option 1: CLI Command (Recommended)
```bash
claude mcp add --transport sse ccadvance http://localhost:8051/sse
```

#### Option 2: Project-wide Configuration
```bash
claude mcp add --transport sse ccadvance http://localhost:8051/sse -s project
```
Creates a `.mcp.json` file that can be committed to version control.

#### Option 3: User-wide Configuration
```bash
claude mcp add --transport sse ccadvance http://localhost:8051/sse -s user
```

#### Option 4: JSON Configuration
```bash
claude mcp add-json ccadvance '{"transport":"sse","serverUrl":"http://localhost:8051/sse"}'
```

### Available Tools in Claude Code

Once integrated, Claude Code will have access to these tools:

| Tool | Purpose | Use Case |
|------|---------|----------|
| `crawl_single_page` | Index a single webpage | Quick documentation page indexing |
| `smart_crawl_url` | Auto-detect and crawl entire sites | Index full documentation sites |
| `crawl_documentation_site` | Specialized JS-heavy site crawling | Modern framework docs (React, Vue, etc.) |
| `get_available_sources` | List indexed domains | Discover what's available to search |
| `perform_rag_query` | Search indexed content | Find relevant documentation |
| `search_code_examples` | Find specific code snippets | Locate implementation examples |

## Key Features for Development Workflows

- **Smart Documentation Crawling** - Handles modern JavaScript-heavy docs sites
- **Code Example Extraction** - Automatically identifies and indexes code snippets
- **Contextual Search** - Find relevant information based on your development context
- **Source Filtering** - Target searches to specific documentation domains
- **Real-time Integration** - Works seamlessly within Claude Code sessions

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- [Supabase Account](https://supabase.com/) for vector storage
- [OpenAI API Key](https://platform.openai.com/api-keys) for embeddings

### Setup Steps

1. **Clone and build**
   ```bash
   git clone https://github.com/vincenthopf/CCAdvance.git
   cd CCAdvance
   docker build -t ccadvance --build-arg PORT=8051 .
   ```

2. **Configure database**
   - Create a Supabase project
   - Run the SQL from `crawled_pages.sql` in your SQL Editor

3. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

4. **Start the server**
   ```bash
   docker run --env-file .env -p 8051:8051 ccadvance
   ```

## Configuration

### Essential Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key (for enhanced features) | No |

### RAG Enhancement Options

| Variable | Description | Recommended |
|----------|-------------|-------------|
| `USE_HYBRID_SEARCH` | Combine vector and keyword search | `true` |
| `USE_RERANKING` | Improve result relevance | `true` |
| `USE_AGENTIC_RAG` | Extract and index code examples | `true` |
| `USE_CONTEXTUAL_EMBEDDINGS` | Enhanced semantic understanding | `false` |

### Recommended Configuration for Claude Code

```env
# Core settings
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051

# API keys (required)
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key

# Enhanced RAG for development
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=true
USE_CONTEXTUAL_EMBEDDINGS=false

# JavaScript crawling (for modern docs)
ENABLE_JS_CRAWLING=true
ENABLE_JS_HOOKS=true
```

## Advanced Configuration

### Browser Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `BROWSER_HEADLESS` | Run browser in headless mode | `true` |
| `VIEWPORT_WIDTH` | Browser viewport width | `1920` |
| `VIEWPORT_HEIGHT` | Browser viewport height | `1080` |

### Timeout Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CRAWL_TIME_SECONDS` | Maximum total crawl time | `2400` (40 min) |
| `MAX_RECURSIVE_CRAWL_TIME_SECONDS` | Maximum recursive crawl time | `1800` (30 min) |

## Alternative Integration Methods

### For Other MCP Clients

If you're not using Claude Code, you can integrate with other MCP clients:

#### SSE Transport
```json
{
  "mcpServers": {
    "ccadvance": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

#### Stdio Transport
```json
{
  "mcpServers": {
    "ccadvance": {
      "command": "python",
      "args": ["path/to/CCAdvance/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_key",
        "SUPABASE_URL": "your_url",
        "SUPABASE_SERVICE_KEY": "your_key"
      }
    }
  }
}
```

## Technical Details

### RAG Strategies

**Hybrid Search**: Combines vector similarity with keyword matching for comprehensive results.

**Reranking**: Uses cross-encoder models to improve result relevance and ordering.

**Agentic RAG**: Extracts code examples (≥300 characters) and creates specialized search indices.

**Contextual Embeddings**: Enhances chunks with document context for better semantic understanding.

### JavaScript Crawling Features

- **Dynamic Content Loading** - Executes JavaScript for SPA sites
- **Lazy Loading Support** - Auto-scrolls to trigger content loading
- **Session Persistence** - Maintains browser state across pages
- **Navigation Extraction** - Follows documentation site navigation

### Rate Limiting & Reliability

- **Exponential Backoff** - Up to 10 retry attempts
- **Graceful Degradation** - Never fails completely on API limits
- **Jitter Prevention** - Avoids thundering herd problems

## Contributing

CCAdvance is built on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/). Contributions are welcome for improving Claude Code integration, crawling capabilities, and RAG strategies.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Enhance your Claude Code development workflow</strong><br>
  Index any documentation and search it contextually within Claude Code
</div>