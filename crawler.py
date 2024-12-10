from crawlee.playwright import PlaywrightCrawler
from crawlee import Configuration
from typing import List, Dict, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup

class WebPage(BaseModel):
    url: str
    content: str
    title: str
    metadata: Dict[str, Any] = {}

class CrawlResult:
    def __init__(self):
        self.pages: List[WebPage] = []

async def crawl_website(start_url: str, max_pages: int = 5) -> List[WebPage]:
    result = CrawlResult()
    
    async def handle_page(context):
        """Handle each crawled page"""
        page = context.page
        request = context.request
        
        try:
            # Wait for the content to load
            await page.wait_for_load_state("networkidle")
            
            # Get the fully rendered HTML content
            html_content = await page.content()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select('script, style, nav, footer, iframe, .cookie-banner, .ad'):
                element.decompose()
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Get metadata
            metadata = {
                'headers': dict(await page.request.all_headers()),
                'status': page.request.response.status,
                'description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else None,
                'keywords': soup.find('meta', {'name': 'keywords'})['content'] if soup.find('meta', {'name': 'keywords'}) else None,
            }
            
            # Store the page
            result.pages.append(WebPage(
                url=request.url,
                content=text_content,
                title=await page.title(),
                metadata=metadata
            ))
            
            # Automatically enqueue links from the same domain
            await context.enqueue_links({
                'strategy': 'same-domain',  # Only follow links from the same domain
                'transformRequestFunction': lambda req: {
                    **req,
                    'userData': {'depth': request.userData.get('depth', 0) + 1}
                }
            })
            
        except Exception as e:
            print(f"Error processing {request.url}: {str(e)}")
    
    # Configure the crawler
    config = Configuration(
        max_requests=max_pages,
        max_request_retries=2,
    )
    
    crawler = PlaywrightCrawler(
        config,
        async_request_handler=handle_page,
        playwright_options={
            'headless': True,
            'stealth': True,
        },
        max_concurrent_requests=2,
        request_handler_timeout=30000,  # 30 seconds
    )
    
    # Start the crawl with initial URL and depth of 0
    await crawler.run([{
        'url': start_url,
        'userData': {'depth': 0}
    }])
    
    return result.pages 