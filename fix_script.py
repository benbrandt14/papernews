import re

with open("papernews/extract.py", "r") as f:
    extract_py = f.read()

extract_py = extract_py.replace('downloaded = trafilatura.fetch_url(url)', '''from trafilatura.settings import use_config

    config = use_config()
    config.set("DEFAULT", "MAX_FILE_SIZE", "5242880")  # 5MB limit
    config.set("DEFAULT", "TIMEOUT", "5")

    downloaded = trafilatura.fetch_url(url, config=config)''')

with open("papernews/extract.py", "w") as f:
    f.write(extract_py)

with open("papernews/render.py", "r") as f:
    render_py = f.read()

# Fix 1: _STRICT_MATH_RE
render_py = render_py.replace('_TYPST_REPLACE = {', '''_STRICT_MATH_RE = re.compile(r"\\$\\$[^\\$]+?\\$\\$|\\$(?!\\s)[^$\\n]+?(?<!\\s)\\$")

_TYPST_REPLACE = {''')

render_py = render_py.replace('''def typst_escape(s) -> str:
    if s is None:
        return ""
    res = "".join(_TYPST_REPLACE.get(c, c) for c in str(s))
    # Unconditionally escape all hashtags so titles like "#1" don't crash the compiler
    return res.replace("#", r"\\#")''', '''def typst_escape(s) -> str:
    if s is None:
        return ""
    s = str(s)
    out = []
    last_end = 0
    for m in _STRICT_MATH_RE.finditer(s):
        prefix = s[last_end : m.start()]
        prefix_escaped = "".join(_TYPST_REPLACE.get(c, c) for c in prefix).replace(
            "#", r"\\#"
        )
        out.append(prefix_escaped)

        math_text = m.group(0)
        math_escaped = "".join(
            _TYPST_REPLACE.get(c, c) if c != "$" else c for c in math_text
        ).replace("#", r"\\#")
        out.append(math_escaped)
        last_end = m.end()

    suffix = s[last_end:]
    suffix_escaped = "".join(_TYPST_REPLACE.get(c, c) for c in suffix).replace(
        "#", r"\\#"
    )
    out.append(suffix_escaped)

    return "".join(out)''')

render_py = render_py.replace('_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\\s*\\n?(.*?)```", re.DOTALL)', '_FENCE_RE = re.compile(r"^(`{3,})[a-zA-Z0-9_+\\-]*\\s*?\\n(.*?)\\n\\1(?=\\s|$)", re.DOTALL | re.MULTILINE)')

render_py = render_py.replace('processed_images = []', 'processed_images: list[str | tuple[str, float, int]] = []')
render_py = render_py.replace('current_valid_group = []', 'current_valid_group: list[tuple[str, float, int]] = []')

render_py = render_py.replace('''                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = response.read()

                        if data.startswith(b"\\xff\\xd8"):''', '''                    with urllib.request.urlopen(req, timeout=5) as response:
                        data = b""
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            data += chunk
                            if len(data) > 5 * 1024 * 1024:
                                raise ValueError("File exceeds 5MB limit")

                        if data.startswith(b"\\xff\\xd8"):''')

render_py = render_py.replace('''    text = _strip_leading_metadata(text)

    blocks: list[str] = []

    def stash_code(m: re.Match) -> str:
        blocks.append(m.group(1))
        return f"\\x00CB{len(blocks) - 1}\\x00"''', '''    text = _strip_leading_metadata(text)

    # Strip invisible control characters (except standard whitespace)
    text = re.sub(
        r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f\\u200b\\u200c\\u200d\\ufeff]", "", text
    )

    blocks: list[str] = []

    def stash_code(m: re.Match) -> str:
        blocks.append(m.group(2))
        return f"\\x00CB{len(blocks) - 1}\\x00\\n"''')

with open("papernews/render.py", "w") as f:
    f.write(render_py)
