import re

text = "In `Transformer` models, computation = 2 * parameters. Here is *bold* and _italic_ and an unclosed _ in a_b_c. Also **bold** and __italic__. And _ space _."

_STRONG_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_EMPH_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_STRONG_US_RE = re.compile(r"__(?!\s)(.+?)(?<!\s)__")
# For underscore, require word boundaries outside
_EMPH_US_RE = re.compile(r"(?<![a-zA-Z0-9_])_(?!\s)(.+?)(?<!\s)_(?![a-zA-Z0-9_])")

def safe_format(text):
    bits = []

    def stash_strong(m):
        bits.append(f" #strong[{m.group(1)}] ") # Note: adding spaces or just wrapping
        return f"\x00TYP{len(bits)-1}\x00"

    def stash_emph(m):
        bits.append(f" #emph[{m.group(1)}] ")
        return f"\x00TYP{len(bits)-1}\x00"

    stashed = text
    stashed = _STRONG_RE.sub(stash_strong, stashed)
    stashed = _EMPH_RE.sub(stash_emph, stashed)
    stashed = _STRONG_US_RE.sub(stash_strong, stashed)
    stashed = _EMPH_US_RE.sub(stash_emph, stashed)

    # Normally we escape here
    stashed = stashed.replace("*", r"\*").replace("_", r"\_")

    def expand_typ(m):
        return bits[int(m.group(1))].strip()

    stashed = re.sub(r"\x00TYP(\d+)\x00", expand_typ, stashed)
    return stashed

print(safe_format(text))
