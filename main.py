import asyncio
import os
from typing import List
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup
import markdown
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from exa_py import Exa
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EXA_API_KEY = os.getenv('EXA_API_KEY')

# initialize exa
exa = Exa(EXA_API_KEY)

class WebPage(BaseModel):
    url: str
    content: str

class Source(BaseModel):
    url: str
    method: str  # 'initial_crawl', 'search', or 'additional_crawl'
    relevance: str

class Report(BaseModel):
    summary: str
    key_points: List[str]
    recommendations: List[str]
    sources: List[Source]

async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def html_to_markdown(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return markdown.markdown(soup.get_text())

async def crawl_site(start_url: str, max_pages: int = 5) -> List[WebPage]:
    pages = []
    visited = set()
    to_visit = {start_url}
    
    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()
            if url in visited:
                continue
                
            print(f"Crawling: {url}")
            html_content = await fetch_page(session, url)
            if html_content:
                visited.add(url)
                markdown_content = html_to_markdown(html_content)
                pages.append(WebPage(url=url, content=markdown_content))
                
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(url, link['href'])
                    if new_url.startswith(start_url) and new_url not in visited:
                        to_visit.add(new_url)
    
    return pages


model = OpenAIModel('gpt-4o', api_key=OPENAI_API_KEY)
# Create the agent
agent = Agent(
    model=model,
    result_type=Report,
    system_prompt="""
     You are a professional restaurant chain researcher with the ability to both analyze provided web content 
    and search for additional relevant information. Your tasks:
    
    1. Analyze the provided web content from crawled pages
    2. Use the search_web and crawl_additional_site tools to find additional relevant information when needed
    3. Create a comprehensive report that combines both sources of information
    
    Your goal is to compile a detailed research report tailored for our sales team to prepare for demos and sales pitches. Include a brief overview of the company’s mission, products/services, industry, and key stats like founding year, headquarters, and size. Highlight major competitors, market trends, and challenges in their sector. Summarize recent news, such as partnerships or leadership changes, and describe their target customers, use cases, and unique selling points. Provide insights from their website, including tools or solutions offered, along with social media activity, reviews, and testimonials. Finally, identify potential sales angles where our solutions align with their needs, and cite sources or provide links for verification where applicable.
    """
)

# Global variable to store crawled pages
crawled_pages = []

@agent.tool_plain
def get_page_content(page_number: int, total_pages: int) -> str:
    """Get content from a specific page number"""
    if not (0 <= page_number < len(crawled_pages)):
        return f"Invalid page number. Please choose between 0 and {len(crawled_pages)-1}"
    return f"URL: {crawled_pages[page_number].url}\nContent: {crawled_pages[page_number].content[:1500]}..."

@agent.tool_plain
def search_web(query: str, num_results: int = 3) -> str:
    """Search the web for relevant information.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 3)
    """
    try:
        print("search query: ", query)
        results = exa.search_and_contents(
            query,
            use_autoprompt=True,
            num_results=5,
            text=True,
        )    
        
        return str(results)
    except Exception as e:
        print(e)
        return f"Search failed: {str(e)}"

@agent.tool_plain
async def crawl_additional_site(url: str, max_pages: int = 10) -> str:
    """Crawl a specific website and return its content.
    
    Args:
        url: The website URL to crawl
        max_pages: Maximum number of pages to crawl (default: 5)
    """
    try:
        pages = []
        visited = set()
        to_visit = {url}
        
        async with aiohttp.ClientSession() as session:
            while to_visit and len(visited) < max_pages:
                current_url = to_visit.pop()
                if current_url in visited:
                    continue
                    
                print(f"Additional crawl: {current_url}")
                
                try:
                    async with session.get(current_url) as response:
                        if response.status != 200:
                            continue
                        html_content = await response.text()
                        
                        # Parse and clean content
                        soup = BeautifulSoup(html_content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text_content = soup.get_text(separator='\n', strip=True)
                        pages.append({
                            "url": current_url,
                            "content": text_content[:1500]  # Limit content length
                        })
                        # print(text_content[:1500])
                        visited.add(current_url)
                        
                        # Find more links
                        for link in soup.find_all('a', href=True):
                            new_url = urljoin(current_url, link['href'])
                            if (new_url.startswith(url) and 
                                new_url not in visited and 
                                len(visited) < max_pages):
                                to_visit.add(new_url)
                except Exception as e:
                    print(f"Error crawling {current_url}: {e}")
                    continue
        
        # Format results
        if not pages:
            return f"No content found for {url}"
        
        formatted_results = []
        for page in pages:
            formatted_results.append(
                f"URL: {page['url']}\n"
                f"Content: {page['content']}...\n"
            )
        
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Crawling failed: {str(e)}"

async def generate_report(prompt: str):
    """Separate async function for generating the report"""
    return await agent.run(prompt)

async def main(url: str):
    global crawled_pages
    
    print("Starting crawl...")
    crawled_pages = await crawl_site(url)
    print(f"Crawled {len(crawled_pages)} pages")
    
    # Prepare initial content summary
    initial_content = "\n\n".join([
        f"Page {i}:\nURL: {page.url}\nContent Preview: {page.content[:1000]}..."
        for i, page in enumerate(crawled_pages)
    ])
    
    prompt = f"""
    I have crawled {len(crawled_pages)} pages from {url}. Here is the initial content:

    {initial_content}

    Please analyze this content and follow these steps:
    1. Review the above content and use get_page_content tool if you need the full content of any page
    2. Use search_web to find additional relevant information about the company
    3. If you find particularly relevant websites, use crawl_additional_site to analyze them in depth

    Create a detailed report that synthesizes all sources, including:
    - A comprehensive summary of the company, their products/services, and market position
    - Key points about their mission, target customers, and unique selling points
    - Data-backed recommendations for sales approaches
    - Analysis of any recent news, partnerships, or industry trends
    
    Please be thorough but focus on information that would be most valuable for our sales team.
    """
    
    print("Generating report...")
    result = await generate_report(prompt)
    
    print("\nReport:")
    print("=======")
    print(f"\nSummary:\n{result.data.summary}")
    print("\nKey Points:")
    for point in result.data.key_points:
        print(f"- {point}")
    print("\nRecommendations:")
    for rec in result.data.recommendations:
        print(f"- {rec}")
    print("\nSources:")
    for source in result.data.sources:
        print(f"- {source.url} ({source.method})")

def run():
    """Wrapper function to run the async main"""
    url = input("Enter the URL to analyze: ")
    asyncio.run(main(url))

if __name__ == "__main__":
    run()
