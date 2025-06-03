<div align="center">
  <img src="./public/Logo.svg" alt="CCAdvance Logo" width="120" height="120">
  
  # CCAdvance
  
  **Advanced Features for Claude Code**
</div>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models
- **Dual LLM Provider Support** with OpenAI for embeddings and OpenRouter for chat completions
- **Aggressive Rate Limit Handling** with exponential backoff to ensure reliable operation
- **Improved Parallel Processing** with optimized worker counts for better stability

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Enhanced JavaScript Support**: Executes JavaScript for dynamic content loading with customizable wait conditions
- **Documentation Site Optimization**: Specialized handling for JavaScript-heavy documentation sites with navigation extraction
- **Lazy Loading Support**: Automatically scrolls pages to trigger lazy-loaded content and images
- **Session Management**: Maintains browser sessions for efficient multi-page crawling on JS-heavy sites
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously with optimized worker counts
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process
- **Dual LLM Provider Support**: Uses OpenAI for cost-effective embeddings and OpenRouter for flexible chat completions
- **Robust Rate Limit Handling**: Implements aggressive exponential backoff (up to 10 retries) to handle API rate limits
- **Enhanced Error Recovery**: Never fails completely on rate limits, always returns graceful fallbacks
- **Flexible Model Configuration**: Easily switch between different models for embeddings and chat completions

## Tools

The server provides essential web crawling and search tools:

### Core Tools (Always Available)

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
   - Now with enhanced JavaScript support for dynamic content
   - Automatically handles lazy-loaded images and expandable sections
   
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
   - Enhanced with JavaScript execution for documentation sites
   - Maintains session state for better performance on JS-heavy sites
   
3. **`crawl_documentation_site`**: Specialized tool for crawling JavaScript-heavy documentation sites
   - Handles dynamic navigation and content loading
   - Extracts and follows documentation navigation links
   - Maintains browser session for efficient multi-page crawling
   - Perfect for sites like Apple Developer Documentation, React docs, etc.
   
4. **`get_available_sources`**: Get a list of all available sources (domains) in the database

5. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

### Conditional Tools

6. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)
- [OpenRouter API key](https://openrouter.ai/keys) (optional, for chat completions with more model options)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# OpenRouter API Configuration (optional, for chat completions)
OPENROUTER_API_KEY=your_openrouter_api_key

# Model Configuration
MODEL_CHOICE=openai/gpt-4.1-nano  # OpenRouter model for chat completions
EMBEDDING_MODEL=text-embedding-3-small  # OpenAI model for embeddings

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Browser Configuration (for JavaScript-heavy sites)
BROWSER_HEADLESS=true  # Run browser in headless mode
BROWSER_VERBOSE=false  # Enable verbose browser logging
VIEWPORT_WIDTH=1920    # Browser viewport width in pixels
VIEWPORT_HEIGHT=1080   # Browser viewport height in pixels

# JavaScript Crawling Configuration
ENABLE_JS_CRAWLING=true  # Enable JavaScript execution for documentation sites
ENABLE_JS_HOOKS=true     # Enable browser hooks for complex interactions
```

### New Features in Latest Update

#### **Enhanced JavaScript Crawling Support**
The server now includes advanced capabilities for crawling JavaScript-heavy documentation sites:
- **Dynamic Content Loading**: Executes JavaScript to load content that appears after page load
- **Lazy Loading Support**: Automatically scrolls pages to trigger lazy-loaded images and content
- **Navigation Extraction**: Extracts and follows documentation navigation links
- **Session Persistence**: Maintains browser sessions for efficient multi-page crawling
- **Expandable Sections**: Automatically expands collapsed content sections
- **Custom Wait Conditions**: Waits for specific elements to ensure content is fully loaded

Perfect for crawling modern documentation sites like:
- Apple Developer Documentation
- React/Vue/Angular docs
- MDN Web Docs
- Any JavaScript-powered documentation site

#### **Dual LLM Provider Support**
The server now supports using different LLM providers for different tasks:
- **OpenAI** for embeddings: More cost-effective and reliable for embedding generation
- **OpenRouter** for chat completions: Access to a wider variety of models with better pricing

This separation allows you to optimize costs while maintaining flexibility in model selection.

#### **Enhanced Rate Limit Handling**
- Implements aggressive exponential backoff with up to 10 retry attempts
- Base delay of 2 seconds with exponential increase up to 5 minutes
- Never fails completely - always returns graceful fallbacks
- Includes jitter to prevent thundering herd problems

#### **Optimized Parallel Processing**
- Reduced concurrent workers from 10 to 3 to avoid rate limits
- Better stability when processing large documentation sites
- Maintains performance while preventing API throttling

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `MODEL_CHOICE`) to generate enriched context that gets embedded alongside the chunk content. Now with robust rate limit handling to ensure reliable operation.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing (uses OpenRouter if configured).
- **Enhancement**: Now includes aggressive retry logic to handle rate limits gracefully.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search. Enhanced with robust rate limit handling.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example (uses OpenRouter if configured).
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.
- **Enhancement**: Parallel processing optimized to 3 workers to avoid rate limits while maintaining performance.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

### Browser Configuration Options

The server includes several configuration options to control browser behavior for JavaScript-heavy sites:

#### **BROWSER_HEADLESS** (default: true)
Controls whether the browser runs in headless mode. Set to `false` for debugging to see what the browser is doing.

#### **BROWSER_VERBOSE** (default: false)
Enables verbose logging from the browser. Useful for troubleshooting crawling issues.

#### **VIEWPORT_WIDTH** / **VIEWPORT_HEIGHT** (default: 1920x1080)
Sets the browser viewport size. Larger viewports can help with sites that hide content on smaller screens.

#### **ENABLE_JS_CRAWLING** (default: true)
Master switch for JavaScript execution features. When enabled, the crawler will:
- Execute JavaScript to load dynamic content
- Scroll pages to trigger lazy loading
- Wait for content to appear before capturing

#### **ENABLE_JS_HOOKS** (default: true)
Enables advanced browser hooks for complex interactions like:
- Accepting cookie banners
- Expanding collapsed sections
- Custom page initialization

### Recommended Configurations

**For general documentation RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For fast, basic RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

**For JavaScript-heavy documentation sites:**
```
BROWSER_HEADLESS=true
ENABLE_JS_CRAWLING=true
ENABLE_JS_HOOKS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## Usage Examples

Here are some examples of how to use the tools provided by this MCP server:

### Crawling a JavaScript-Heavy Documentation Site

Use the specialized `crawl_documentation_site` tool for modern documentation sites:

```
# Crawl Apple's SwiftUI documentation
crawl_documentation_site(url="https://developer.apple.com/documentation/swiftui/", max_pages=50)

# Crawl React documentation with navigation following
crawl_documentation_site(url="https://react.dev/", follow_nav_links=true)
```

### Basic Web Crawling

For simpler sites or when you want more control:

```
# Crawl a single page
crawl_single_page(url="https://example.com/blog/post")

# Smart crawl that auto-detects the URL type
smart_crawl_url(url="https://example.com/sitemap.xml", max_depth=3)
```

### Searching Crawled Content

After crawling, search the indexed content:

```
# Get available sources first
get_available_sources()

# Search with source filtering
perform_rag_query(query="SwiftUI navigation", source="developer.apple.com")

# Search for code examples (requires USE_AGENTIC_RAG=true)
search_code_examples(query="useState hook example", source_id="react.dev")
```

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers
