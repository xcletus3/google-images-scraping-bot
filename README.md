# Educational Image Scraper
Author
CletusXavier - CodexLiber
cletusxavier.2110@protonmail.com
[Your GitHub Profile (optional)]

## Overview
This is an educational web scraping script demonstrating how to download images from Google Images while respecting web scraping best practices.

## ⚠️ Legal Disclaimer
- This script is for EDUCATIONAL PURPOSES ONLY
- Users MUST:
  - Comply with website's robots.txt rules
  - Respect rate limits
  - Obtain permission before scraping
  - Follow copyright and data protection laws

## Features
- Compliant robots.txt checking
- Rate limiting
- Image size and content type validation
- Logging of download activities
- Configurable scraping parameters

## Prerequisites
- Python 3.8+
- Chrome Browser installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xcletus3/google-images-scraping-bot.git
cd google-images-scraping-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from pathlib import Path
from official_scraper_bot import ScraperConfig, RateLimitedScraper

# Configure scraper
config = ScraperConfig(
    download_path=Path("./downloaded_images"),
    rate_limit=2.0,
    user_agent="YourProjectName/1.0"
)

# Initialize and use scraper
scraper = RateLimitedScraper(config)
downloaded_files = scraper.scrape_images(
    search_query="nature landscapes", 
    max_images=10
)
```

## Configuration Options
- `download_path`: Directory to save images
- `rate_limit`: Delay between requests
- `user_agent`: Custom user agent string
- `max_size_mb`: Maximum image file size
- `retry_attempts`: HTTP request retry attempts
- `backoff_factor`: Exponential backoff for retries
- `timeout`: Request timeout duration

## Logging
Logs are saved in `scraper.log` with details about downloads and errors.

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - See LICENSE file for details

## Disclaimer
This is an educational tool. Always use web scraping responsibly and ethically.
