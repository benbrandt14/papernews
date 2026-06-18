import subprocess
import re
import jinja2
from pathlib import Path
from typing import List
from datetime import datetime
from papernews.models import ArticleChunk

def sanitize_typst(text: str) -> str:
    if not text: return ""
    text = str(text)
    text = text.replace('#', r'\#').replace('@', r'\@')
    text = re.sub(r'\$(?=\d)', r'\\$', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'#link("\2")[\1]', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    text = re.sub(r'\[\s*\]', '', text) 
    text = re.sub(r'</?[a-zA-Z]+>', '', text)
    return text

def build_pdf(chunks: List[ArticleChunk], output_filename: str = "papernews.pdf"):
    print("Orchestrating layout regions...")

    # The Agnostic Layout Grid
    regions = {
        "index": {},         # Dictionary grouped by category
        "cover_feature": [], # List of prominent items
        "sidebar": [],       # List of smaller sidebar items
        "interior": {}       # Dictionary grouped by category
    }

    # Group by Region
    for chunk in chunks:
        # Sanitize immediately
        chunk.title = sanitize_typst(chunk.title)
        chunk.summary = sanitize_typst(chunk.summary)
        chunk.body_markdown = sanitize_typst(chunk.body_markdown)
        
        # Route index and interior items into categorical groupings
        if chunk.region in ["index", "interior"]:
            cat = chunk.category.title()
            if cat not in regions[chunk.region]:
                regions[chunk.region][cat] = []
            regions[chunk.region][cat].append(chunk)
        else:
            regions[chunk.region].append(chunk)

    # Initialize Jinja
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        block_start_string="((*", block_end_string="*))",
        variable_start_string="(((", variable_end_string=")))",
        comment_start_string="((#", comment_end_string="#))",
    )
    
    # Generic Helpers
    env.filters['typst'] = sanitize_typst
    env.filters['typst_url'] = lambda x: str(x) if x else ""
    env.filters['typst_body'] = sanitize_typst

    template = env.get_template("template.typ.j2")
    
    # Pass the abstracted regions to the template
    typst_markup = template.render(
        regions=regions,
        date=datetime.now().strftime("%A, %B %d, %Y")
    )

    build_dir = Path(".build")
    build_dir.mkdir(exist_ok=True)
    typst_file = build_dir / "daily_issue.typ"
    pdf_file = build_dir / output_filename
    typst_file.write_text(typst_markup)

    try:
        subprocess.run(["typst", "compile", str(typst_file), str(pdf_file)], check=True)
        print(f"Success! Decoupled PDF compiled to {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"Typst compilation failed: {e}")