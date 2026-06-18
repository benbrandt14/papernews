import subprocess
import re
import jinja2
from pathlib import Path
from typing import List
from datetime import datetime
from papernews.models import ArticleChunk

def sanitize_typst(text: str) -> str:
    """Universal string sanitizer to prevent Typst compiler crashes."""
    if not text: return ""
    text = str(text)
    text = text.replace('#', r'\#').replace('@', r'\@')
    text = re.sub(r'\$(?=\d)', r'\\$', text)
    
    # Safely extract markdown links
    links = []
    def link_repl(match):
        links.append(f'#link("{match.group(2)}")[{match.group(1)}]')
        return f'__LINK_{len(links)-1}__'
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', link_repl, text)
    
    # Escape ALL stray brackets to prevent unclosed delimiter crashes
    text = text.replace('[', r'\[').replace(']', r'\]')
    
    # Re-inject the safe links
    for i, link in enumerate(links):
        text = text.replace(f'__LINK_{i}__', link)
        
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    text = re.sub(r'</?[a-zA-Z]+>', '', text)
    return text

def build_pdf(chunks: List[ArticleChunk], output_filename: str = "papernews.pdf"):
    print("Orchestrating layout regions...")

    regions = {"index": {}, "cover_feature": [], "sidebar": [], "interior": {}}

    for chunk in chunks:
        chunk.title = sanitize_typst(chunk.title)
        chunk.summary = sanitize_typst(chunk.summary)
        chunk.body_markdown = sanitize_typst(chunk.body_markdown)
        
        if chunk.region in ["index", "interior"]:
            cat = chunk.category.title()
            if cat not in regions[chunk.region]:
                regions[chunk.region][cat] = []
            regions[chunk.region][cat].append(chunk)
        else:
            regions[chunk.region].append(chunk)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        block_start_string="((*", block_end_string="*))",
        variable_start_string="(((", variable_end_string=")))",
    )
    
    env.filters['typst'] = sanitize_typst
    env.filters['typst_url'] = lambda x: str(x) if x else ""
    env.filters['typst_body'] = sanitize_typst

    template = env.get_template("template.typ.j2")
    typst_markup = template.render(regions=regions, date=datetime.now().strftime("%A, %B %d, %Y"))

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