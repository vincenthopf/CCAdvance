"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from playwright.async_api import Page, BrowserContext

from utils import (
    get_supabase_client, 
    add_documents_to_supabase, 
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration with enhanced JavaScript support
    browser_config = BrowserConfig(
        headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
        verbose=os.getenv("BROWSER_VERBOSE", "false").lower() == "true",
        # Add viewport size for better rendering
        viewport={
            "width": int(os.getenv("VIEWPORT_WIDTH", "1920")),
            "height": int(os.getenv("VIEWPORT_HEIGHT", "1080"))
        }
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (code, context_before, context_after)
        
    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)

def get_js_code_for_documentation_sites() -> List[str]:
    """
    Get JavaScript code snippets for handling common documentation site patterns.
    
    Returns:
        List of JavaScript code snippets to execute
    """
    return [
        # Scroll to trigger lazy loading
        """
        // Smooth scroll to bottom to trigger lazy loading
        const scrollToBottom = async () => {
            const scrollHeight = document.documentElement.scrollHeight;
            const step = window.innerHeight;
            let currentPosition = 0;
            
            while (currentPosition < scrollHeight) {
                window.scrollTo(0, currentPosition);
                currentPosition += step;
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        };
        scrollToBottom();
        """,
        
        # Expand collapsed sections if they exist
        """
        // Expand all collapsible sections
        const expandButtons = document.querySelectorAll('[aria-expanded="false"], .collapsed, .expand-button');
        expandButtons.forEach(button => {
            if (button.click) button.click();
        });
        """,
        
        # Click "Show more" or "Load more" buttons
        """
        // Click any "Show more" or "Load more" buttons
        const loadMoreButtons = document.querySelectorAll(
            'button:contains("more"), button:contains("More"), ' +
            'a:contains("more"), a:contains("More"), ' +
            '.load-more, .show-more'
        );
        loadMoreButtons.forEach(button => {
            if (button.click) button.click();
        });
        """
    ]

def get_wait_condition_for_documentation() -> str:
    """
    Get a JavaScript wait condition for documentation sites.
    
    Returns:
        JavaScript function that returns true when content is loaded
    """
    return """() => {
        // Wait for main content to load
        const content = document.querySelector('main, .content, #content, article, .documentation');
        if (!content || content.innerText.length < 100) return false;
        
        // Check if there are code blocks (common in documentation)
        const codeBlocks = document.querySelectorAll('pre, code, .highlight');
        
        // Check if navigation is loaded (for documentation sites)
        const nav = document.querySelector('nav, .navigation, .sidebar');
        
        // Consider loaded if we have content and either code blocks or navigation
        return content && (codeBlocks.length > 0 || nav);
    }"""

def create_enhanced_crawler_config(
    url: str,
    enable_js: bool = True,
    wait_for_dynamic_content: bool = True,
    scroll_page: bool = True,
    session_id: Optional[str] = None,
    js_only: bool = False
) -> CrawlerRunConfig:
    """
    Create an enhanced crawler configuration for JavaScript-heavy sites.
    
    Args:
        url: The URL being crawled
        enable_js: Whether to enable JavaScript execution
        wait_for_dynamic_content: Whether to wait for dynamic content
        scroll_page: Whether to scroll the page for lazy loading
        session_id: Optional session ID for multi-step crawling
        js_only: Whether to reuse existing session without navigation
        
    Returns:
        Enhanced CrawlerRunConfig
    """
    config_params = {
        "cache_mode": CacheMode.BYPASS,
        "stream": False,
    }
    
    # Add JavaScript handling for documentation sites
    if enable_js and is_documentation_site(url):
        if scroll_page:
            config_params["js_code"] = get_js_code_for_documentation_sites()
        
        if wait_for_dynamic_content:
            config_params["wait_for"] = f"js:{get_wait_condition_for_documentation()}"
            config_params["page_timeout"] = 60000  # 60 seconds
            config_params["delay_before_return_html"] = 2.0  # Wait 2 seconds before capturing
        
        # Enable lazy loading support
        config_params["wait_for_images"] = True
        config_params["scan_full_page"] = scroll_page
        config_params["scroll_delay"] = 0.5
    
    # Add session management
    if session_id:
        config_params["session_id"] = session_id
        config_params["js_only"] = js_only
    
    return CrawlerRunConfig(**config_params)

def is_documentation_site(url: str) -> bool:
    """
    Check if a URL is likely a documentation site.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL appears to be a documentation site
    """
    doc_patterns = [
        'docs', 'documentation', 'developer', 'api',
        'guide', 'tutorial', 'reference', 'manual'
    ]
    
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in doc_patterns)

async def setup_page_hooks(crawler: AsyncWebCrawler) -> None:
    """
    Set up hooks for handling complex JavaScript interactions.
    
    Args:
        crawler: The AsyncWebCrawler instance
    """
    async def before_retrieve_html(page: Page, context: BrowserContext, **kwargs) -> Page:
        """Hook to execute before retrieving HTML - useful for final scrolls."""
        try:
            # Final scroll to ensure all content is loaded
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(1000)  # Wait 1 second
        except Exception as e:
            print(f"Error in before_retrieve_html hook: {e}")
        return page
    
    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs) -> Page:
        """Hook for page initialization - useful for setting up the page."""
        try:
            # Set a larger viewport for documentation sites
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            # Accept cookies if there's a banner (common on documentation sites)
            try:
                cookie_button = await page.query_selector('button:has-text("Accept"), button:has-text("OK"), .cookie-accept')
                if cookie_button:
                    await cookie_button.click()
            except:
                pass
        except Exception as e:
            print(f"Error in on_page_context_created hook: {e}")
        return page
    
    # Attach hooks to the crawler
    if hasattr(crawler, 'crawler_strategy'):
        crawler.crawler_strategy.set_hook("before_retrieve_html", before_retrieve_html)
        crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Set up hooks for enhanced JavaScript handling
        if os.getenv("ENABLE_JS_HOOKS", "true").lower() == "true":
            await setup_page_hooks(crawler)
        
        # Configure the crawl with enhanced JavaScript support
        enable_js = os.getenv("ENABLE_JS_CRAWLING", "true").lower() == "true"
        run_config = create_enhanced_crawler_config(
            url=url,
            enable_js=enable_js,
            wait_for_dynamic_content=True,
            scroll_page=True
        )
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                total_word_count += meta.get("word_count", 0)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
            update_source_info(supabase_client, source_id, source_summary, total_word_count)
            
            # Add documentation chunks to Supabase (AFTER source exists)
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    # Process code examples in parallel (reduced workers to avoid rate limits)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
                    
                    # Add code examples to Supabase
                    add_code_examples_to_supabase(
                        supabase_client, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas
                    )
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if code_blocks else 0,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}
        
        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)
                
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Update source information for each unique source FIRST (before inserting documents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))
        
        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)
        
        # Add documentation chunks to Supabase (AFTER sources exist)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
        
        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            all_code_blocks = []
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)
                
                if code_blocks:
                    # Process code examples in parallel (reduced workers to avoid rate limits)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
            
            # Add all code examples to Supabase
            if code_examples:
                add_code_examples_to_supabase(
                    supabase_client, 
                    code_urls, 
                    code_chunk_numbers, 
                    code_examples, 
                    code_summaries, 
                    code_metadatas,
                    batch_size=batch_size
                )
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples),
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_documentation_site(ctx: Context, url: str, max_pages: int = 50, follow_nav_links: bool = True) -> str:
    """
    Crawl a documentation site with advanced JavaScript handling and navigation following.
    
    This tool is specifically designed for JavaScript-heavy documentation sites that load content
    dynamically. It handles:
    - Dynamic content loading with JavaScript
    - Lazy-loaded images and code blocks
    - Expandable/collapsible sections
    - JavaScript-based navigation
    - Session persistence for better performance
    
    Args:
        ctx: The MCP server provided context
        url: URL of the documentation site to crawl
        max_pages: Maximum number of pages to crawl (default: 50)
        follow_nav_links: Whether to follow navigation links (default: True)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Set up hooks for documentation sites
        await setup_page_hooks(crawler)
        
        # Create a session for the documentation site
        parsed_url = urlparse(url)
        session_id = f"doc_session_{parsed_url.netloc}"
        
        # First, crawl the main page to discover navigation structure
        initial_config = create_enhanced_crawler_config(
            url=url,
            enable_js=True,
            wait_for_dynamic_content=True,
            scroll_page=True,
            session_id=session_id
        )
        
        # Add custom JavaScript to extract navigation links
        nav_extraction_js = """
        // Extract navigation links from common documentation patterns
        const navLinks = [];
        const selectors = [
            'nav a', '.nav a', '.navigation a', '.sidebar a',
            '.toc a', '.table-of-contents a', '[role="navigation"] a',
            '.docs-nav a', '.documentation-nav a'
        ];
        
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(link => {
                const href = link.href;
                const text = link.textContent.trim();
                if (href && text && href.startsWith(window.location.origin)) {
                    navLinks.push({href, text});
                }
            });
        });
        
        // Store in window for retrieval
        window.__navLinks = navLinks;
        """
        
        initial_config.js_code = initial_config.js_code or []
        if isinstance(initial_config.js_code, str):
            initial_config.js_code = [initial_config.js_code]
        initial_config.js_code.append(nav_extraction_js)
        
        # Crawl the initial page
        result = await crawler.arun(url=url, config=initial_config)
        
        urls_to_crawl = [url]
        crawled_results = []
        
        if result.success:
            crawled_results.append({'url': url, 'markdown': result.markdown})
            
            # Extract navigation links if follow_nav_links is enabled
            if follow_nav_links:
                try:
                    # Try to get navigation links from the page
                    nav_links_config = CrawlerRunConfig(
                        session_id=session_id,
                        js_only=True,
                        js_code="JSON.stringify(window.__navLinks || [])",
                        cache_mode=CacheMode.BYPASS
                    )
                    nav_result = await crawler.arun(url=url, config=nav_links_config)
                    
                    if nav_result.success and nav_result.html:
                        try:
                            # Parse navigation links
                            nav_links = json.loads(nav_result.html)
                            for link in nav_links[:max_pages - 1]:  # Limit to max_pages
                                if link['href'] not in urls_to_crawl:
                                    urls_to_crawl.append(link['href'])
                        except json.JSONDecodeError:
                            print("Could not parse navigation links")
                except Exception as e:
                    print(f"Error extracting navigation links: {e}")
        
        # Crawl remaining URLs using the session
        for i, next_url in enumerate(urls_to_crawl[1:], 1):
            if i >= max_pages:
                break
                
            try:
                next_config = create_enhanced_crawler_config(
                    url=next_url,
                    enable_js=True,
                    wait_for_dynamic_content=True,
                    scroll_page=True,
                    session_id=session_id,
                    js_only=False  # Navigate to new pages
                )
                
                result = await crawler.arun(url=next_url, config=next_config)
                if result.success and result.markdown:
                    crawled_results.append({'url': next_url, 'markdown': result.markdown})
            except Exception as e:
                print(f"Error crawling {next_url}: {e}")
        
        # Clean up session
        if hasattr(crawler, 'crawler_strategy'):
            try:
                await crawler.crawler_strategy.kill_session(session_id)
            except Exception as e:
                print(f"Error cleaning up session: {e}")
        
        # Process and store results
        if not crawled_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Store in Supabase (similar to smart_crawl_url)
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        source_id = parsed_url.netloc
        total_word_count = 0
        
        for doc in crawled_results:
            doc_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=5000)
            
            for i, chunk in enumerate(chunks):
                urls.append(doc_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = doc_url
                meta["source"] = source_id
                meta["crawl_type"] = "documentation"
                metadatas.append(meta)
                
                total_word_count += meta.get("word_count", 0)
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {doc['url']: doc['markdown'] for doc in crawled_results}
        
        # Update source information
        source_summary = extract_source_summary(source_id, crawled_results[0]['markdown'][:5000])
        update_source_info(supabase_client, source_id, source_summary, total_word_count)
        
        # Add to Supabase
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": "documentation",
            "pages_crawled": len(crawled_results),
            "chunks_stored": chunk_count,
            "total_word_count": total_word_count,
            "source_id": source_id,
            "urls_crawled": [doc['url'] for doc in crawled_results][:10] + (["..."] if len(crawled_results) > 10 else [])
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering 
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Query the sources table directly
        result = supabase_client.from_('sources')\
            .select('*')\
            .order('source_id')\
            .execute()
        
        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append({
                    "source_id": source.get("source_id"),
                    "summary": source.get("summary"),
                    "total_words": source.get("total_words"),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                })
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE
            keyword_query = supabase_client.from_('crawled_pages')\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')
            
            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq('source_id', source)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.
    
    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "error": "Code example extraction is disabled. Perform a normal RAG search."
        }, indent=2)
    
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # Import the search function from utils
            from utils import search_code_examples as search_code_examples_impl
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = supabase_client.from_('code_examples')\
                .select('id, url, chunk_number, content, summary, metadata, source_id')\
                .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')
            
            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq('source_id', source_id)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            from utils import search_code_examples as search_code_examples_impl
            
            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Text files usually don't need JavaScript handling
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel with enhanced JavaScript support.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Set up hooks once for all crawls
    if os.getenv("ENABLE_JS_HOOKS", "true").lower() == "true" and urls:
        # Check if any URL is a documentation site
        if any(is_documentation_site(url) for url in urls):
            await setup_page_hooks(crawler)
    
    # Create configurations for each URL
    configs = []
    enable_js = os.getenv("ENABLE_JS_CRAWLING", "true").lower() == "true"
    
    for url in urls:
        config = create_enhanced_crawler_config(
            url=url,
            enable_js=enable_js and is_documentation_site(url),
            wait_for_dynamic_content=True,
            scroll_page=True
        )
        configs.append(config)
    
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )
    
    # If all URLs need the same config, we can use a single config
    if all(is_documentation_site(url) == is_documentation_site(urls[0]) for url in urls):
        results = await crawler.arun_many(urls=urls, config=configs[0], dispatcher=dispatcher)
    else:
        # Otherwise, crawl each URL individually with its specific config
        results = []
        for url, config in zip(urls, configs):
            try:
                result = await crawler.arun(url=url, config=config)
                if result.success and result.markdown:
                    results.append(result)
            except Exception as e:
                print(f"Error crawling {url}: {e}")
    
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth with enhanced JavaScript support.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    # Set up hooks once for documentation sites
    if os.getenv("ENABLE_JS_HOOKS", "true").lower() == "true" and start_urls:
        if any(is_documentation_site(url) for url in start_urls):
            await setup_page_hooks(crawler)
    
    enable_js = os.getenv("ENABLE_JS_CRAWLING", "true").lower() == "true"
    
    # Create session ID for documentation sites to maintain state
    session_id = None
    if enable_js and start_urls and is_documentation_site(start_urls[0]):
        session_id = f"doc_session_{urlparse(start_urls[0]).netloc}"
    
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    # Extract base path from the first start URL to use as prefix filter
    base_path_prefix = None
    if start_urls:
        parsed = urlparse(start_urls[0])
        # Get the directory path (remove any file name)
        path = parsed.path
        if path and not path.endswith('/'):
            # If the path doesn't end with /, treat it as a directory
            path = path + '/'
        # Use the scheme, netloc, and directory path for filtering
        base_path_prefix = f"{parsed.scheme}://{parsed.netloc}{path}"

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        # Create configurations for each URL
        configs = []
        for url in urls_to_crawl:
            config = create_enhanced_crawler_config(
                url=url,
                enable_js=enable_js and is_documentation_site(url),
                wait_for_dynamic_content=True,
                scroll_page=True,
                session_id=session_id if is_documentation_site(url) else None,
                js_only=depth > 0 and session_id is not None  # Reuse session after first depth
            )
            configs.append(config)
        
        # Crawl URLs
        if all(is_documentation_site(url) == is_documentation_site(urls_to_crawl[0]) for url in urls_to_crawl):
            # All URLs are similar, use same config
            results = await crawler.arun_many(urls=urls_to_crawl, config=configs[0], dispatcher=dispatcher)
        else:
            # Mixed URL types, crawl individually
            results = []
            for url, config in zip(urls_to_crawl, configs):
                try:
                    result = await crawler.arun(url=url, config=config)
                    results.append(result)
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
        
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    
                    # Filter URLs to only include those under the base path
                    if base_path_prefix and not next_url.startswith(base_path_prefix):
                        continue
                    
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls
    
    # Clean up session if created
    if session_id and hasattr(crawler, 'crawler_strategy'):
        try:
            await crawler.crawler_strategy.kill_session(session_id)
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")

    return results_all

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())