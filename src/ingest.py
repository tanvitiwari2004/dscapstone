import json
import re
from pathlib import Path

SRC = Path("data/sources/ba_liquids_and_restrictions.txt")
OUT = Path("data/chunks.jsonl")

SOURCE_URL = "https://www.britishairways.com/content/information/baggage-essentials/liquids-and-restrictions"
CAPTURED_ON = "2025-12-14"

SECTION_RE = re.compile(r"^===\s*(.+?)\s*===$")

def clean_line(s: str) -> str:
    s = s.replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def chunk_text(text: str, max_chars=1200, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max(1, max_chars - overlap)
    return chunks

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source file: {SRC}")

    lines = SRC.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Parse into sections based on headings like === Heading ===
    sections = []
    current_section = "Unsorted"
    buffer = []

    # Skip metadata header until content starts (optional but helps)
    started = False
    for raw in lines:
        line = raw.strip()

        if not started:
            if "CONTENT STARTS BELOW" in line or "CONTENT STARTS" in line:
                started = True
            continue

        m = SECTION_RE.match(line)
        if m:
            # flush previous section
            if buffer:
                sections.append((current_section, "\n".join(buffer)))
                buffer = []
            current_section = m.group(1).strip()
            continue

        cl = clean_line(line)
        if cl:
            buffer.append(cl)

    if buffer:
        sections.append((current_section, "\n".join(buffer)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        chunk_id = 0
        for section_name, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            for ch in chunk_text(section_text):
                obj = {
                    "chunk_id": f"ba_lr_{chunk_id:03d}",
                    "section": section_name,
                    "source": SOURCE_URL,
                    "captured_on": CAPTURED_ON,
                    "text": ch.strip()
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"✅ Wrote {chunk_id} chunks to {OUT}")

if __name__ == "__main__":
    main()
