import asyncio
import os
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from exa_py import Exa
from dotenv import load_dotenv
from crawler import crawl_website, WebPage

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EXA_API_KEY = os.getenv('EXA_API_KEY')

# initialize exa
exa = Exa(EXA_API_KEY)

class Source(BaseModel):
    url: str
    method: str
    relevance: str

class Report(BaseModel):
    summary: str
    key_points: List[str]
    recommendations: List[str]
    sources: List[Source]

model = OpenAIModel('gpt-4o', api_key=OPENAI_API_KEY)
agent = Agent(
    model=model,
    result_type=Report,
    system_prompt="""
    You are a professional restaurant chain researcher with the ability to both analyze provided web content 
    and search for additional relevant information. Your tasks:
    
    1. Analyze the provided web content from crawled pages
    2. Use the search_web tool to find additional relevant information when needed
    3. Create a comprehensive report that combines both sources of information
    
    Your goal is to compile a detailed research report tailored for our sales team to prepare for demos and sales pitches.
    Include company information, market position, competitors, trends, and potential sales angles.
    """
)

# Global variable to store crawled pages
crawled_pages: List[WebPage] = []

@agent.tool_plain
def get_page_content(page_number: int) -> str:
    """Get content from a specific page number"""
    if not (0 <= page_number < len(crawled_pages)):
        return f"Invalid page number. Please choose between 0 and {len(crawled_pages)-1}"
    page = crawled_pages[page_number]
    return f"URL: {page.url}\nTitle: {page.title}\nContent: {page.content[:1500]}..."

@agent.tool_plain
def search_web(query: str, num_results: int = 3) -> str:
    """Search the web for relevant information."""
    try:
        print(f"Searching: {query}")
        results = exa.search_and_contents(
            query,
            use_autoprompt=True,
            num_results=num_results,
            text=True,
        )    
        return str(results)
    except Exception as e:
        print(e)
        return f"Search failed: {str(e)}"

async def generate_report(prompt: str):
    """Generate the report"""
    return await agent.run(prompt)

async def main(url: str):
    global crawled_pages
    
    print("Starting crawl...")
    crawled_pages = await crawl_website(url, max_pages=5)
    print(f"Crawled {len(crawled_pages)} pages")
    
    # Prepare content summary
    initial_content = "\n\n".join([
        f"Page {i}:\nURL: {page.url}\nTitle: {page.title}\nContent Preview: {page.content[:1000]}..."
        for i, page in enumerate(crawled_pages)
    ])
    
    prompt = f"""
    I have crawled {len(crawled_pages)} pages from {url}. Here is the initial content:

    {initial_content}

    Please analyze this content and:
    1. Review the content (use get_page_content for full page content if needed)
    2. Use search_web to find additional relevant information
    3. Create a detailed report synthesizing all sources

    Focus on information valuable for our sales team, including:
    - Company overview (mission, products/services, market position)
    - Target customers and unique selling points
    - Recent news and industry trends
    - Potential sales approaches
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
    """Entry point"""
    url = input("Enter the URL to analyze: ")
    asyncio.run(main(url))

if __name__ == "__main__":
    run()
