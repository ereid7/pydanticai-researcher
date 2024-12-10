from typing import List, Dict, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup
from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext
from datetime import timedelta
from crawlee import Request


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

    crawler = PlaywrightCrawler(
        max_requests_per_crawl=max_pages,
        headless=True,
        request_handler_timeout=timedelta(
            seconds=30
        ),  # Changed from milliseconds to timedelta
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f"Processing {context.request.url} ...")

        try:
            # Wait for the content to load
            await context.page.wait_for_load_state("networkidle")

            # Get the fully rendered HTML content
            html_content = await context.page.content()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements
            for element in soup.select(
                "script, style, nav, footer, iframe, .cookie-banner, .ad"
            ):
                element.decompose()

            # Extract text content
            text_content = soup.get_text(separator="\n", strip=True)

            # Get metadata
            metadata = {
                "headers": dict(await context.page.request.all_headers()),
                "status": context.page.request.response.status,
                "description": (
                    soup.find("meta", {"name": "description"})["content"]
                    if soup.find("meta", {"name": "description"})
                    else None
                ),
                "keywords": (
                    soup.find("meta", {"name": "keywords"})["content"]
                    if soup.find("meta", {"name": "keywords"})
                    else None
                ),
            }

            # Store the page
            result.pages.append(
                WebPage(
                    url=context.request.url,
                    content=text_content,
                    title=await context.page.title(),
                    metadata=metadata,
                )
            )

            # Automatically enqueue links from the same domain
            await context.enqueue_links(
                {
                    "strategy": "same-domain",  # Only follow links from the same domain
                    "transformRequestFunction": lambda req: {
                        **req,
                        "userData": {
                            "depth": context.request.userData.get("depth", 0) + 1
                        },
                    },
                }
            )

        except Exception as e:
            context.log.error(f"Error processing {context.request.url}: {str(e)}")

    # Start the crawl with initial URL and depth of 0
    initial_request = Request(url=start_url, user_data={"depth": 0})
    await crawler.run([initial_request])

    return result.pages
