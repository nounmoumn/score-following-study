"""Extract external project/data links from the published SKY-Piano site bundle."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results/p3_external_metadata/skypiano_site.js'
raw = source.read_text(encoding='utf-8')
urls = sorted(set(re.findall(r'https?://[^\s"\'<>\\]+', raw)))
for url in urls:
    if any(word in url.lower() for word in ('zenodo', 'github', 'huggingface', 'drive.google', 'dataset', 'skypiano')):
        print(url[:350])
