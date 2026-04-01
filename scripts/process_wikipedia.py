#!/usr/bin/env python3
"""
Process Wikipedia Chinese dataset for training.
"""

import os
import re
import json
import bz2
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET
from tqdm import tqdm
import argparse


class WikipediaProcessor:
    """Process Wikipedia XML dump to extract clean text."""
    
    def __init__(self, 
                 min_text_length: int = 50,
                 max_text_length: int = 10000):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # Patterns to clean
        self.clean_patterns = [
            (r'\{\{.*?\}\}', ''),          # Remove templates {{...}}
            (r'\[\[.*?\]\]', ''),          # Remove links [[...]]
            (r'\<.*?\>', ''),              # Remove HTML tags <...>
            (r'\={2,}.*?\={2,}', ''),      # Remove headers ==...==
            (r'\*+', ''),                   # Remove bullet points
            (r'\#REDIRECT', ''),           # Remove redirect markers
            (r'Category:.*', ''),          # Remove categories
            (r'File:.*', ''),              # Remove file references
            (r'https?://\S+', ''),         # Remove URLs
            (r'\s+', ' '),                 # Normalize whitespace
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia article text."""
        
        # Apply cleaning patterns
        for pattern, replacement in self.clean_patterns:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        
        # Remove references [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove special characters but keep Chinese
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9a-zA-Z\s，。！？；：""''、\.\,\!\?\;\:\"\'\-\(\)\[\]]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid for training."""
        
        if not text:
            return False
        
        if len(text) < self.min_text_length:
            return False
        
        if len(text) > self.max_text_length:
            return False
        
        # Check Chinese character ratio
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars < len(text) * 0.5:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^#REDIRECT',
            r'^#重定向',
            r'{{',
            r'}}',
            r'\[\[',
            r'\]\]',
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text):
                return False
        
        return True
    
    def parse_wiki_xml(self, xml_path: str) -> Iterator[tuple]:
        """Parse Wikipedia XML dump file."""
        
        print(f"Parsing {xml_path}...")
        
        # Get file size for progress
        file_size = Path(xml_path).stat().st_size
        print(f"File size: {file_size / 1024 / 1024 / 1024:.2f} GB")
        
        # Check if file is bz2 compressed
        if xml_path.endswith('.bz2'):
            print("Note: Parsing compressed file directly...")
            print("Recommend: Decompress first for better progress tracking")
            open_func = lambda p: bz2.open(p, 'rt', encoding='utf-8')
        else:
            open_func = lambda p: open(p, 'r', encoding='utf-8')
        
        with open_func(xml_path) as f:
            # Use iterparse for memory efficiency
            context = ET.iterparse(f, events=('start', 'end'))
            
            page_count = 0
            checked_count = 0
            
            # Create progress bar
            pbar = tqdm(total=None, desc="Parsing XML", unit="pages", mininterval=1.0)
            
            for event, elem in context:
                if event == 'end' and elem.tag == 'page':
                    checked_count += 1
                    pbar.update(1)
                    
                    # Update postfix every 100 pages to avoid slowdown
                    if checked_count % 100 == 0:
                        pbar.set_postfix({'valid': page_count}, refresh=False)
                    
                    # Extract title
                    title_elem = elem.find('title')
                    title = title_elem.text if title_elem is not None else ''
                    
                    # Extract text
                    text_elem = elem.find('.//text')
                    text = text_elem.text if text_elem is not None else ''
                    
                    # Clean and validate
                    cleaned = ''
                    if text:
                        cleaned = self.clean_text(text)
                        if self.is_valid_text(cleaned):
                            yield title, cleaned
                            page_count += 1
                    
                    # Clear element to save memory
                    elem.clear()
                    
                    # Show first few valid articles for debugging
                    if page_count <= 3 and page_count > 0:
                        pbar.write(f"\nSample article {page_count}:")
                        pbar.write(f"  Title: {title}")
                        pbar.write(f"  Text length: {len(cleaned)}")
                        pbar.write(f"  Preview: {cleaned[:80]}...")
            
            pbar.write(f"\nTotal: {checked_count} pages checked, {page_count} valid articles found")
            pbar.close()
    
    def process_wikipedia(self, 
                         xml_path: str,
                         output_path: str,
                         max_articles: Optional[int] = None) -> int:
        """Process Wikipedia dump and save clean text."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Processing Wikipedia Chinese Dataset")
        print(f"{'='*60}")
        print(f"Input:  {xml_path}")
        print(f"Output: {output_path}")
        print(f"Min length: {self.min_text_length}")
        print(f"Max length: {self.max_text_length}")
        
        article_count = 0
        total_chars = 0
        
        # Create progress bar for writing
        write_pbar = tqdm(desc="Writing articles", unit="articles")
        
        with open(output_file, 'w', encoding='utf-8', buffering=1) as f:  # Line buffering
            for title, text in self.parse_wiki_xml(xml_path):
                # Write to file
                f.write(text + '\n')
                f.flush()  # Force flush immediately
                
                article_count += 1
                total_chars += len(text)
                
                # Update progress bar
                write_pbar.update(1)
                write_pbar.set_postfix({
                    'chars': f'{total_chars/1000000:.1f}M',
                    'size': f'{output_file.stat().st_size/1024/1024:.1f}MB'
                })
                
                # Limit number of articles
                if max_articles and article_count >= max_articles:
                    write_pbar.write(f"\nReached max articles limit: {max_articles}")
                    break
        
        write_pbar.close()
        
        print(f"\n{'='*60}")
        print("Processing Complete")
        print(f"{'='*60}")
        print(f"Total articles: {article_count:,}")
        print(f"Total characters: {total_chars:,} ({total_chars/1000000:.1f}M)")
        print(f"Average length: {total_chars/article_count:.0f} chars/article")
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size/1024/1024:.1f} MB")
        
        # Save statistics
        stats = {
            'total_articles': article_count,
            'total_characters': total_chars,
            'avg_length': total_chars / article_count if article_count > 0 else 0,
            'min_length': self.min_text_length,
            'max_length': self.max_text_length,
        }
        
        stats_file = output_file.parent / f"{output_file.stem}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return article_count


def process_with_wikiextractor(bz2_path: str, output_dir: str) -> str:
    """Alternative: Use WikiExtractor tool."""
    
    import subprocess
    
    print("Using WikiExtractor tool...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if wikiextractor is installed
    try:
        subprocess.run(['python', '-m', 'wikiextractor', '--version'], 
                      capture_output=True, check=True)
    except:
        print("Installing wikiextractor...")
        subprocess.run(['pip', 'install', 'wikiextractor'], check=True)
    
    # Run WikiExtractor
    cmd = [
        'python', '-m', 'wikiextractor.WikiExtractor',
        '--output', str(output_dir),
        '--processes', '4',
        '--no_templates',
        '--filter_disambig_pages',
        '--min_text_length', '50',
        bz2_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Combine extracted files
    combined_file = output_dir / 'combined.txt'
    print(f"Combining extracted files to {combined_file}...")
    
    with open(combined_file, 'w', encoding='utf-8') as outf:
        for txt_file in output_dir.rglob('wiki_*'):
            if txt_file.is_file():
                with open(txt_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        if line.strip() and not line.startswith('<'):
                            outf.write(line)
    
    return str(combined_file)


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia dataset")
    parser.add_argument("--input", type=str, 
                       default="data/zhwiki-latest-pages-articles.xml.bz2",
                       help="Wikipedia XML dump file (.bz2)")
    parser.add_argument("--output", type=str, 
                       default="data/wiki_processed.txt",
                       help="Output text file")
    parser.add_argument("--max-articles", type=int, default=None,
                       help="Maximum number of articles to extract")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=10000,
                       help="Maximum text length")
    parser.add_argument("--use-wikiextractor", action="store_true",
                       help="Use WikiExtractor tool instead")
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nPlease download the file first:")
        print(f"  wget -c {args.input}")
        return
    
    if args.use_wikiextractor:
        output = process_with_wikiextractor(args.input, "data/wiki_extracted")
        print(f"\nExtracted text saved to: {output}")
    else:
        processor = WikipediaProcessor(
            min_text_length=args.min_length,
            max_text_length=args.max_length
        )
        processor.process_wikipedia(
            args.input,
            args.output,
            max_articles=args.max_articles
        )


if __name__ == "__main__":
    main()