import os
import re
import concurrent.futures
from collections import Counter
from pathlib import Path
import sys
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Regex to find the Tags in a .osu file's [Metadata] section
TAGS_REGEX = re.compile(r'^Tags\s*:\s*(.*)$', re.MULTILINE | re.IGNORECASE)

def extract_tags(file_path: Path) -> str | None:
    """Reads a .osu file and extracts its tags."""
    try:
        # Reading the first 4KB should be plenty for the Metadata section
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4096)
            match = TAGS_REGEX.search(content)
            if match:
                return match.group(1).strip().lower()
    except (IOError, UnicodeDecodeError):
        pass
    return None

def main():
    osu_dir = PROJECT_ROOT / "data" / "raw"
    if not osu_dir.exists():
        print(f"Directory {osu_dir} does not exist.")
        return

    osu_files = list(osu_dir.glob("*.osu"))
    
    console = Console()
    console.print(f"Found [bold cyan]{len(osu_files)}[/bold cyan] .osu files.")

    if not osu_files:
        console.print("[bold red]No .osu files found in data/raw.[/bold red]")
        return

    # Multithreaded extraction
    worker_threads = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as executor:
        results = list(tqdm(executor.map(extract_tags, osu_files), total=len(osu_files), desc="Extracting tags"))
        
    # Filter out empty tags
    all_tags_docs = [tags for tags in results if tags]
    
    if not all_tags_docs:
        console.print("[bold red]No tags found in any files.[/bold red]")
        return

    console.print(f"Extracted tags from [bold green]{len(all_tags_docs)}[/bold green] files.")

    # Basic Metrics
    tag_counts_per_map = [len(doc.split()) for doc in all_tags_docs]
    avg_tags = np.mean(tag_counts_per_map)
    median_tags = np.median(tag_counts_per_map)
    max_tags = np.max(tag_counts_per_map)
    
    all_words = []
    for doc in all_tags_docs:
        all_words.extend(doc.split())
    
    word_counts = Counter(all_words)
    unique_tags = len(word_counts)

    # TF-IDF Analysis
    console.print("\n[bold yellow]Calculating TF-IDF scores...[/bold yellow]")
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b') 
    tfidf_matrix = vectorizer.fit_transform(all_tags_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Mean TF-IDF score for each tag across all documents
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    tfidf_series = pd.Series(mean_tfidf, index=feature_names)
    top_tfidf = tfidf_series.sort_values(ascending=False).head(50)

    # Display Summary Metrics
    summary_table = Table(title="Tag Summary Metrics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Beatmaps", str(len(all_tags_docs)))
    summary_table.add_row("Total Unique Tags", str(unique_tags))
    summary_table.add_row("Average Tags per Map", f"{avg_tags:.2f}")
    summary_table.add_row("Median Tags per Map", f"{median_tags:.1f}")
    summary_table.add_row("Max Tags on a Map", str(max_tags))
    console.print(summary_table)

    # Display Frequency Results
    table_freq = Table(title="Top 50 Tags by Frequency")
    table_freq.add_column("Tag", style="cyan")
    table_freq.add_column("Frequency", justify="right", style="green")
    for tag, count in word_counts.most_common(50):
        table_freq.add_row(tag, str(count))
    console.print(table_freq)

    # Display TF-IDF Results
    table_tfidf = Table(title="Top 50 Tags by Mean TF-IDF Score")
    table_tfidf.add_column("Tag", style="cyan")
    table_tfidf.add_column("Mean TF-IDF", justify="right", style="magenta")
    for tag, score in top_tfidf.items():
        table_tfidf.add_row(tag, f"{score:.4f}")
    console.print(table_tfidf)

if __name__ == "__main__":
    main()
