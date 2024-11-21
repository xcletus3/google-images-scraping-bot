"""
Educational Image Scraping Demo
-----------------------------
Purpose: Educational demonstration of web scraping techniques
Author: CletusXavier {/CodexLiber}

LEGAL DISCLAIMER AND USAGE RESTRICTIONS:
--------------------------------------
1. This code is provided for EDUCATIONAL PURPOSES ONLY to demonstrate web scraping techniques.
2. Users must:
   - Comply with each website's robots.txt rules and Terms of Service
   - Respect rate limits and implement proper delays
   - Only download images they have rights to use
   - Follow website's terms of service regarding automated access
   - Obtain explicit permission before scraping any website
   - Respect intellectual property rights and copyright laws
   - Comply with GDPR and other relevant data protection laws

3. The author accepts no responsibility for:
   - Misuse of this code
   - Any violations of terms of service
   - Copyright infringement
   - Any other legal issues arising from use of this code

4. Production Use:
   - This code is NOT intended for production use
   - For production applications, use official APIs instead
   - Consider using Google's Custom Search API for actual image searches

5. Rate Limiting:
   - Default rate limiting is implemented but may need adjustment
   - Users should implement appropriate delays
   - Monitor and respect server response codes

Copyright (c) 2024 CletusXavier {/CodexLiber}
Licensed under the MIT License - see LICENSE file for details
"""

from __future__ import annotations
from typing import Optional, Dict, List, ClassVar
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from pathlib import Path
import logging
import time
import random
import os
import mimetypes
from datetime import datetime
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass


@dataclass
class ScraperConfig:
    """Configuration settings for the scraper."""
    download_path: Path
    rate_limit: float
    user_agent: str
    max_size_mb: float = 10.0
    retry_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        if self.rate_limit < 0:
            raise ValueError("Rate limit must be non-negative")
        if self.max_size_mb <= 0:
            raise ValueError("Maximum size must be positive")
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")
        if self.backoff_factor < 0:
            raise ValueError("Backoff factor must be non-negative")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


class RobotsChecker:
    """Handles robots.txt checking with caching and rate limiting."""

    def __init__(self, user_agent: str) -> None:
        """Initialize the RobotsChecker.

        Args:
            user_agent: User agent string for robots.txt requests
        """
        self.user_agent = user_agent
        self._cache: Dict[str, RobotFileParser] = {}
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_robots_url(url: str) -> str:
        """Generate robots.txt URL from base URL.

        Args:
            url: Base URL to generate robots.txt URL from

        Returns:
            Complete robots.txt URL
        """
        parsed = urlparse(url)
        return urljoin(f"{parsed.scheme}://{parsed.netloc}", "robots.txt")

    def _get_parser(self, domain: str) -> Optional[RobotFileParser]:
        """Get or create a robots.txt parser for a domain.

        Args:
            domain: Domain to get parser for

        Returns:
            RobotFileParser instance or None if unable to fetch
        """
        if domain in self._cache:
            return self._cache[domain]

        try:
            parser = RobotFileParser()
            parser.set_url(domain)
            parser.read()
            self._cache[domain] = parser
            return parser
        except Exception as e:
            self.logger.error(f"Failed to fetch robots.txt for {domain}: {e}")
            return None

    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            Boolean indicating if URL can be fetched
        """
        try:
            robots_url = self._get_robots_url(url)
            parser = self._get_parser(robots_url)

            if parser is None:
                self.logger.warning(f"Could not check robots.txt for {url}, assuming disallowed")
                return False

            allowed = parser.can_fetch(self.user_agent, url)
            crawl_delay = float(parser.crawl_delay(self.user_agent))

            if crawl_delay:
                time.sleep(crawl_delay)

            return allowed

        except Exception as e:
            self.logger.error(f"Error checking robots.txt for {url}: {e}")
            return False


class ImageValidator(ABC):
    """Abstract base class for image validation."""

    @abstractmethod
    def validate(self, response: requests.Response) -> None:
        """Validate image response.

        Args:
            response: Response object to validate

        Raises:
            ImageValidationError: If validation fails
        """
        pass


class SizeValidator(ImageValidator):
    """Validates image size."""

    def __init__(self, max_size_mb: float) -> None:
        """Initialize size validator.

        Args:
            max_size_mb: Maximum allowed size in megabytes
        """
        self.max_size_mb = max_size_mb

    def validate(self, response: requests.Response) -> None:
        """Validate image size is within limits.

        Args:
            response: Response object to validate

        Raises:
            ImageValidationError: If image size exceeds limit
        """
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = float(content_length) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                raise ImageValidationError(
                    f"Image size ({size_mb:.2f}MB) exceeds limit of {self.max_size_mb}MB"
                )


class ContentTypeValidator(ImageValidator):
    """Validates image content type."""

    def validate(self, response: requests.Response) -> None:
        """Validate response contains an image.

        Args:
            response: Response object to validate

        Raises:
            ImageValidationError: If content type is not an image
        """
        content_type = response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            raise ImageValidationError(
                f"Invalid content type: {content_type}. Expected image/*"
            )


class RateLimitedScraper:
    """A rate-limited web scraper with robots.txt compliance."""

    # Class constants
    CONTENT_TYPE_MAP: ClassVar[Dict[str, str]] = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp'
    }

    def __init__(self, config: ScraperConfig) -> None:
        """Initialize the scraper with configuration.

        Args:
            config: ScraperConfig instance with scraper settings
        """
        self.config = config
        self.robots_checker = RobotsChecker(config.user_agent)
        self.validators = [
            SizeValidator(config.max_size_mb),
            ContentTypeValidator()
        ]

        self.headers = {
            "User-Agent": config.user_agent,
            "Accept": "image/webp,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        self.session = self._create_session()
        self._setup_logging()
        self._create_directory_structure()

    def _setup_logging(self) -> None:
        """Configure logging with proper format and handlers."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            fh = logging.FileHandler('scraper.log')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic.

        Returns:
            Configured requests Session
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.retry_attempts,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @staticmethod
    def clean_image_url(url: str) -> str:
        """Clean the image URL by removing query parameters.

        Args:
            url: URL to clean

        Returns:
            Cleaned URL string
        """
        return url.split('?')[0]

    def _create_directory_structure(self) -> None:
        """Create directory structure with proper permissions."""
        try:
            self.config.download_path.mkdir(parents=True, exist_ok=True, mode=0o755)
            self.logger.info(f"Created directory: {self.config.download_path}")
        except OSError as e:
            self.logger.error(f"Failed to create directory {self.config.download_path}: {e}")
            raise

    @staticmethod
    def _get_file_extension(response: requests.Response) -> str:
        """Determine file extension from response headers.

        Args:
            response: Response object to get extension from

        Returns:
            File extension string
        """
        content_type = response.headers.get('Content-Type', '').lower()

        if content_type:
            if content_type in RateLimitedScraper.CONTENT_TYPE_MAP:
                return RateLimitedScraper.CONTENT_TYPE_MAP[content_type]

            ext = mimetypes.guess_extension(content_type)
            if ext:
                return ext

        return '.jpg'

    def generate_safe_filename(self, url: str, response: requests.Response) -> str:
        """Generate a safe filename from URL with proper extension.

        Args:
            url: Source URL
            response: Response object for content type detection

        Returns:
            Safe filename string
        """
        original_name = Path(urlparse(url).path).name
        safe_name = re.sub(r'[^\w\-_.]', '', original_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self._get_file_extension(response)

        if '.' in safe_name:
            base_name = safe_name.rsplit('.', 1)[0]
        else:
            base_name = safe_name

        return f"{timestamp}_{base_name}{extension}"

    def download_image(self, url: str) -> Optional[Path]:
        """Download and save an image with validation and error handling.

        Args:
            url: URL of image to download

        Returns:
            Path to downloaded file or None if download failed
        """
        try:
            if not self.robots_checker.can_fetch(url):
                self.logger.info(f"Skipping download - not allowed by robots.txt: {url}")
                return None

            clean_url = self.clean_image_url(url)

            response = self.session.get(
                clean_url,
                headers=self.headers,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()

            # Run all validators
            for validator in self.validators:
                validator.validate(response)

            filename = self.generate_safe_filename(clean_url, response)
            save_path = self.config.download_path / filename

            with save_path.open('wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            os.chmod(save_path, 0o644)
            self.logger.info(f"Successfully downloaded: {save_path}")

            # Implement rate limiting
            time.sleep(random.uniform(self.config.rate_limit, self.config.rate_limit * 1.5))

            return save_path

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None

    def scrape_images(self, search_query: str, max_images: int = 10) -> List[Path]:
        """Search and download images based on query.

        Args:
            search_query: Search term for images
            max_images: Maximum number of images to download

        Returns:
            List of paths to downloaded images
        """
        downloaded_images: List[Path] = []
        chrome_options = Options()
        chrome_options.add_argument('--headless')

        driver = None

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
            self.logger.info(f"Starting search for: {search_query}")

            driver.get(search_url)
            time.sleep(3)

            # Scroll to load more images
            for _ in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            # Find all image links
            image_elements = driver.find_elements(By.CSS_SELECTOR, 'h3.ob5Hkd a')

            for index, image_link in enumerate(image_elements[:max_images]):
                try:
                    # Open image in new tab using Control+Click
                    action_chains = ActionChains(driver)
                    action_chains.key_down(Keys.CONTROL).click(image_link).key_up(Keys.CONTROL).perform()
                    time.sleep(2)

                    # Switch to the newly opened tab
                    driver.switch_to.window(driver.window_handles[-1])

                    # Find the full resolution image
                    image = driver.find_element(By.CSS_SELECTOR, 'div.p7sI2 img.iPVvYb')
                    src_url = image.get_attribute('src')

                    if path := self.download_image(src_url):
                        downloaded_images.append(path)

                except Exception as e:
                    self.logger.error(f"Error processing image {index}: {str(e)}")

                finally:
                    # Close the tab and switch back to main window
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(random.uniform(1, 2))

        except Exception as e:
            self.logger.error(f"Error in scraping process: {str(e)}")

        finally:
            driver.quit()

        return downloaded_images


def main() -> None:
    """Example usage of the scraper."""
    # Configuration
    config = ScraperConfig(
        download_path=Path("./downloaded_images"),
        rate_limit=2.0,
        user_agent="EducationalScraper/1.0 (Educational Project +https://your-website.com)",
        max_size_mb=10.0,
        retry_attempts=3,
        backoff_factor=2.0,
        timeout=10.0
    )

    # Initialize scraper
    scraper = RateLimitedScraper(config)

    # Start scraping
    downloaded_files = scraper.scrape_images(
        search_query="nature landscapes",
        max_images=10
    )

    # Print results
    print(f"\nDownloaded {len(downloaded_files)} images:")
    for file in downloaded_files:
        print(f"- {file}")


if __name__ == '__main__':
    main()
