import httpx
from bs4 import BeautifulSoup

from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent


class URLDataSourceProvider(BaseDataSourceProvider):
    """Provider for URL/web scraping data sources."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def fetch_content(self, source: str) -> DataSourceContent:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                # Parse HTML and extract text
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Get text content
                text = soup.get_text(separator="\n", strip=True)

                # Get title
                title = soup.title.string if soup.title else source

                return DataSourceContent(
                    content=text,
                    source=source,
                    metadata={
                        "title": title,
                        "content_type": content_type,
                        "url": str(response.url),
                    },
                )
            else:
                # Return raw text for non-HTML content
                return DataSourceContent(
                    content=response.text,
                    source=source,
                    metadata={
                        "content_type": content_type,
                        "url": str(response.url),
                    },
                )

    async def validate_source(self, source: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(source, follow_redirects=True)
                return response.status_code == 200
        except Exception:
            return False
