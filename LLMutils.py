"""
LLMutils.py - A collection of utility functions for project documentation and asset management.
These utilities were created to help synthesize the Home Credit Risk project for dissertation submission.

Functions include:
1. Image Extraction (from Markdown and Jupyter Notebooks)
2. Markdown Formatting (Heading transformation, Link updates, Image insertion)
3. Asset Management (Descriptive renaming, Deduplication)
"""

import json
import base64
import os
import glob
import re

def slugify(text):
    """Converts a string to a safe, lowercase alphanumeric slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

# --- 1. IMAGE EXTRACTION ---

def extract_base64_from_markdown(md_file, output_dir):
    """
    Extracts base64-encoded images from a markdown file (usually at the end of the file).
    Written because legacy reports from teammates contained embedded PNGs that needed to be individual assets for the dissertation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(md_file, 'r') as f:
        content = f.readlines()
    for i in range(len(content)):
        line = content[i].strip()
        match = re.search(r'\[image(\d+)\]: <data:image/(\w+);base64,([^>]+)>', line)
        if match:
            img_num, img_ext, img_data = match.groups()
            filename = f"image_{img_num}.{img_ext}"
            with open(os.path.join(output_dir, filename), 'wb') as img_file:
                img_file.write(base64.b64decode(img_data))

def extract_images_from_notebooks(src_dir, output_dir):
    """
    Parses Jupyter Notebooks in the src directory to extract high-resolution PNG outputs.
    Written to recover higher quality images than what was available in the markdown exports.
    Includes context-aware naming based on nearby markdown headers or code comments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    notebooks = glob.glob(os.path.join(src_dir, "*.ipynb"))
    for nb_path in notebooks:
        nb_name = os.path.basename(nb_path).replace(".ipynb", "")
        with open(nb_path, 'r', encoding='utf-8') as f:
            try: nb = json.load(f)
            except: continue
        img_count = 0
        last_markdown = ""
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                if source: last_markdown = "".join(source).split('\n')[0].strip('# ')
            elif cell.get('cell_type') == 'code':
                context = last_markdown or ""
                source_code = "".join(cell.get('source', []))
                first_comment = re.search(r'#\s*(.*)', source_code)
                if first_comment and len(context) < 3: context = first_comment.group(1)
                context_slug = slugify(context)[:30] or "plot"
                for output in cell.get('outputs', []):
                    if 'data' in output and 'image/png' in output['data']:
                        img_data = output['data']['image/png']
                        if isinstance(img_data, list): img_data = "".join(img_data)
                        img_count += 1
                        filename = f"{nb_name}_{context_slug}_{img_count}.png"
                        with open(os.path.join(output_dir, filename), 'wb') as img_file:
                            img_file.write(base64.b64decode(img_data))
                last_markdown = ""

# --- 2. MARKDOWN REFORMATTING ---

def all_caps_to_headers(md_path):
    """
    Scans a markdown file and turns lines consisting only of capital letters into H1 headers.
    Written to format raw PPT text exports into structured markdown sections.
    """
    with open(md_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        orig = line.strip()
        if orig and not orig.startswith('#') and not orig.startswith('<!--'):
            if any(c.isupper() for c in orig) and not any(c.islower() for c in orig):
                line = "# " + line.lstrip()
        new_lines.append(line)
    with open(md_path, 'w') as f:
        f.writelines(new_lines)

def replace_base64_links_with_local(md_path, img_dir_rel):
    """
    Replaces massive base64 image definitions with compact local file paths.
    Written to reduce markdown file size and make it readable after image extraction.
    """
    with open(md_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        match = re.search(r'\[image(\d+)\]: <data:image/(\w+);base64,[^>]+>', line)
        if match:
            num, ext = match.groups()
            new_lines.append(f"[image{num}]: {img_dir_rel}/image_{num}.{ext}\n")
        else:
            new_lines.append(line)
    with open(md_path, 'w') as f:
        f.writelines(new_lines)

def insert_slide_images(md_path, img_dir):
    """
    Context-aware image insertion. Matches H1 headers with image filename prefixes.
    Written to automatically place presentation screenshots under their correct slide text.
    """
    files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    file_map = {}
    for f in files:
        parts = re.split(r'[-\u2013\u2014]', f)
        slug = slugify(parts[0].strip())
        if slug not in file_map: file_map[slug] = []
        file_map[slug].append(f)
    
    with open(md_path, 'r') as f:
        content = f.read()
    slides = re.split(r'(<!-- Slide number: \d+ -->)', content)
    new_slides = []
    for slide in slides:
        if slide.startswith('<!-- Slide number:'):
            new_slides.append(slide)
            continue
        processed_lines = []
        slide_words = slugify(slide)
        for line in slide.split('\n'):
            processed_lines.append(line)
            match = re.match(r'^#\s+(.+)$', line)
            if match:
                h_slug = slugify(match.group(1).strip())
                matches = []
                for f_slug in file_map:
                    if f_slug == h_slug or (len(f_slug) > 5 and f_slug in h_slug) or (len(h_slug) > 5 and h_slug in f_slug):
                        matches.extend(file_map[f_slug])
                if matches:
                    for m in sorted(list(set(matches))):
                        processed_lines.append(f"![{m}]({img_dir}/{m})")
        new_slides.append("\n".join(processed_lines))
    with open(md_path, 'w') as f:
        f.write("".join(new_slides))

# --- 3. POLISHING ---

def deduplicate_images(md_path):
    """
    Ensures every unique image path is mentioned only once in the entire document.
    Cleans captions to remove slide name prefixes (text before hyphen).
    Written to finalize the PPT conversion and prevent redundant visual clutter.
    """
    with open(md_path, 'r') as f:
        lines = f.readlines()
    seen = set()
    new_lines = []
    for line in lines:
        match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            caption, path = match.groups()
            if path in seen: continue
            seen.add(path)
            clean_cap = caption.replace('.png', '')
            if '-' in clean_cap: clean_cap = clean_cap.split('-', 1)[1].strip()
            line = line.replace(f"![{caption}]({path})", f"![{clean_cap}]({path})")
        new_lines.append(line)
    with open(md_path, 'w') as f:
        f.writelines(new_lines)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 4. DATA PLOTTING UTILITIES ---

def plot_external_sources(csv_path, output_path):
    """
    Generates distribution plots for EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3.
    Consolidated from scratch_plot_ext_sources.py to keep specialized plotting logic in one place.
    """
    print("Loading data...")
    df = pd.read_csv(csv_path, usecols=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])
    
    print("Plotting distributions...")
    plt.figure(figsize=(15, 5))
    features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    for i, feature in enumerate(features):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[feature].dropna(), kde=True, bins=50, color='skyblue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("LLMutils loaded. Scripts consolidated.")
