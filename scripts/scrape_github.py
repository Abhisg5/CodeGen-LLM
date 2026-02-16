import os
import requests
import json
from typing import List, Dict
import time
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubScraper:
    def __init__(self, token: str = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN in .env file or pass token to constructor.")
            
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.total_files_downloaded = 0
        self.MAX_FILES_PER_REPO = 10  # Limit files per repository
        self.MAX_FILE_SIZE_KB = 100  # Skip files larger than 100KB
        
        # Load configuration
        config_path = Path('config/github_orgs.json')
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
    def get_user_repos(self, username: str) -> List[Dict]:
        """Get all repositories for a specific user."""
        url = f"{self.base_url}/users/{username}/repos"
        params = {
            'sort': 'updated',
            'direction': 'desc',
            'per_page': 100
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting repositories for user {username}: {e}")
            return []

    def get_org_repos(self, org: str) -> List[Dict]:
        """Get repositories from a specific organization."""
        url = f"{self.base_url}/orgs/{org}/repos"
        params = {
            'sort': 'stars',
            'direction': 'desc',
            'per_page': 100
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting organization repositories: {e}")
            return []

    def get_repository_contents(self, owner: str, repo: str, path: str = '') -> List[Dict]:
        """Get contents of a repository directory."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting repository contents: {e}")
            return []

    def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Get content of a specific file."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            content = response.json()['content']
            
            # Check file size (content is base64 encoded)
            size_kb = len(content) * 3/4 / 1024  # Approximate size in KB
            if size_kb > self.MAX_FILE_SIZE_KB:
                logger.info(f"Skipping {path} - file too large ({size_kb:.1f}KB)")
                return ""
                
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting file content: {e}")
            return ""

    def scrape_repository(self, owner: str, repo: str, output_dir: str):
        """Scrape Python files from a repository with size limits."""
        output_path = Path(output_dir) / f"{owner}_{repo}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_downloaded = 0
        
        def process_directory(path: str = ''):
            nonlocal files_downloaded
            if files_downloaded >= self.MAX_FILES_PER_REPO:
                return
                
            contents = self.get_repository_contents(owner, repo, path)
            
            for item in contents:
                if files_downloaded >= self.MAX_FILES_PER_REPO:
                    return
                    
                if item['type'] == 'dir':
                    process_directory(item['path'])
                elif item['type'] == 'file' and item['name'].endswith('.py'):
                    content = self.get_file_content(owner, repo, item['path'])
                    if content:
                        file_path = output_path / item['path']
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(content)
                        files_downloaded += 1
                        self.total_files_downloaded += 1
                        logger.info(f"Saved {item['path']} ({files_downloaded}/{self.MAX_FILES_PER_REPO} files for this repo)")
                
                # Respect GitHub API rate limits
                time.sleep(0.1)
        
        process_directory()

def main():
    # Create output directory
    output_dir = Path('data/raw/github')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize scraper
    scraper = GitHubScraper()
    
    # Process organizations
    for org in scraper.config['organizations']:
        org_name = org['name']
        org_desc = org['description']
        logger.info(f"Getting repositories from organization: {org_name} ({org_desc})")
        org_repos = scraper.get_org_repos(org_name)
        
        # Process top repositories from each org
        for repo in org_repos[:2]:  # Take top 2 repos from each org
            owner = repo['owner']['login']
            repo_name = repo['name']
            logger.info(f"Scraping organization repository: {owner}/{repo_name}")
            
            try:
                scraper.scrape_repository(owner, repo_name, str(output_dir))
            except Exception as e:
                logger.error(f"Error scraping {owner}/{repo_name}: {e}")
            
            # Respect GitHub API rate limits
            time.sleep(1)
    
    # Process users
    for user in scraper.config['users']:
        username = user['name']
        user_desc = user['description']
        logger.info(f"Getting repositories for user: {username} ({user_desc})")
        user_repos = scraper.get_user_repos(username)
        
        for repo in user_repos:
            owner = repo['owner']['login']
            repo_name = repo['name']
            logger.info(f"Scraping user repository: {owner}/{repo_name}")
            
            try:
                scraper.scrape_repository(owner, repo_name, str(output_dir))
            except Exception as e:
                logger.error(f"Error scraping {owner}/{repo_name}: {e}")
            
            # Respect GitHub API rate limits
            time.sleep(1)
    
    # Process specific repositories
    for repo in scraper.config['repositories']:
        owner = repo['owner']
        repo_name = repo['name']
        repo_desc = repo['description']
        logger.info(f"Scraping specific repository: {owner}/{repo_name} ({repo_desc})")
        
        try:
            scraper.scrape_repository(owner, repo_name, str(output_dir))
        except Exception as e:
            logger.error(f"Error scraping {owner}/{repo_name}: {e}")
        
        # Respect GitHub API rate limits
        time.sleep(1)
    
    logger.info(f"Scraping completed. Downloaded {scraper.total_files_downloaded} files total.")

if __name__ == "__main__":
    main() 