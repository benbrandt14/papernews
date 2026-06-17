import subprocess
from pathlib import Path
from typing import List
import jinja2
import re
from papernews.models import LayoutChunk
from datetime import datetime

def sanitize_typst(text: str) -> str:
    """Sanitizes raw LLM markdown into compiler-safe Typst strings."""
    if not text:
        return ""

    # 1. Escape literal hashtags (prevents "unknown variable" crashes)
    # We do this first so we can safely inject valid Typst '#' commands below.
    text = text.replace('#', r'\#')

    # 2. Escape literal @ symbols (prevents "label does not exist" crashes)
    text = text.replace('@', r'\@')

    # 3. Escape currency dollar signs (prevents unclosed math block crashes)
    # Matches $ followed by a digit, leaving $math$ and $$ blocks alone
    text = re.sub(r'\$(?=\d)', r'\\$', text)

    # 4. Convert Markdown Links to native Typst Links
    # e.g., [Text](URL) -> #link("URL")[Text]
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'#link("\2")[\1]', text)

    # 5. Convert Markdown Bold (**) to Typst Strong (*)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)

    # 6. Clean up stray/empty brackets that cause jarring layout breaks
    text = re.sub(r'\[\s*\]', '', text)

    return text

def build_pdf(chunks: List[LayoutChunk], output_filename: str = "papernews.pdf"):
    """Takes validated LayoutChunks, groups them, and compiles the Typst PDF."""
    print("Preparing chunks for typesetting...")

    # --- Apply universal sanitization to all LLM output ---
    for chunk in chunks:
        chunk.headline = sanitize_typst(chunk.headline)
        chunk.body_markdown = sanitize_typst(chunk.body_markdown)

    # Route chunks to their specific layout buckets
    hero_chunk = next((c for c in chunks if c.template_type == "hero_grid"), None)
    sidebar_chunks = [c for c in chunks if c.template_type == "sidebar_tease"]
    academic_chunks = [c for c in chunks if c.template_type == "academic_digest"]
    standard_chunks = [c for c in chunks if c.template_type == "standard_article"]

    # Fallback: if no hero chunk was generated, promote the highest priority standard chunk
    if not hero_chunk and standard_chunks:
        hero_chunk = standard_chunks.pop(0)

    # Setup Jinja Environment (using custom delimiters to avoid Typst `#` conflicts)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        block_start_string="(((*",
        block_end_string="*)))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
    )
    
    template = env.get_template("template.typ.j2")
    
    # Render the Typst markup
    typst_markup = template.render(
        hero=hero_chunk,
        sidebars=sidebar_chunks,
        academics=academic_chunks,
        standards=standard_chunks,
        run_time=datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
    )

    # Write temporary file and compile
    build_dir = Path(".build")
    build_dir.mkdir(exist_ok=True)

    font_dir = Path(__file__).parent.parent / "fonts"
    
    typst_file = build_dir / "daily_issue.typ"
    pdf_file = build_dir / output_filename
    
    typst_file.write_text(typst_markup)

    print("Compiling Typst PDF...")
    try:
        subprocess.run(
            ["typst", "compile", "--font-path", str(font_dir), str(typst_file), str(pdf_file)], 
            check=True
        )
        print(f"Success! Newspaper compiled to {pdf_file}")
    except FileNotFoundError:
        print("Error: 'typst' CLI not found. Ensure Typst is installed in your uv environment.")
    except subprocess.CalledProcessError as e:
        print(f"Typst compilation failed: {e}")