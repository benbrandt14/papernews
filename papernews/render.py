import subprocess
import re
import jinja2
from pathlib import Path
from typing import List
from datetime import datetime
from papernews.models import ArticleChunk

def sanitize_typst(text: str) -> str:
    if not text: return ""
    text = text.replace('#', r'\#').replace('@', r'\@')
    text = re.sub(r'\$(?=\d)', r'\\$', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'#link("\2")[\1]', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'</?[a-zA-Z]+>', '', text)
    return text

def build_pdf(chunks: List[ArticleChunk], output_filename: str = "papernews.pdf"):
    print("Grouping articles and preparing classic template...")

    articles_by_category = {}
    wiki_events = []
    quote = None

    for chunk in chunks:
        # Sanitize raw LLM outputs
        chunk.title = sanitize_typst(chunk.title)
        chunk.summary = sanitize_typst(chunk.summary)
        chunk.body_markdown = sanitize_typst(chunk.body_markdown)

        # Group by layout type
        if chunk.content_type == "wiki_quote":
            quote = chunk
        elif chunk.content_type == "wiki_event":
            wiki_events.append(chunk)
        else:
            cat = chunk.category.title()
            if cat not in articles_by_category:
                articles_by_category[cat] = []
            articles_by_category[cat].append(chunk)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        block_start_string="(((*", block_end_string="*)))",
        variable_start_string="(((", variable_end_string=")))",
    )
    
    template = env.get_template("template.typ.j2")
    typst_markup = template.render(
        articles_by_category=articles_by_category,
        wiki_events=wiki_events,
        quote=quote,
        run_time=datetime.now().strftime("%A, %B %d, %Y")
    )

    build_dir = Path(".build")
    build_dir.mkdir(exist_ok=True)
    typst_file = build_dir / "daily_issue.typ"
    pdf_file = build_dir / output_filename
    typst_file.write_text(typst_markup)

    try:
        subprocess.run(["typst", "compile", str(typst_file), str(pdf_file)], check=True)
        print(f"Success! Classic Magazine compiled to {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"Typst compilation failed: {e}")