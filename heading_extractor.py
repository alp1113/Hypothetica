import re
from docling.document_converter import DocumentConverter


def clean_heading_text(text, is_title=False):
    """Clean heading text by removing content after colons and limiting length."""
    if not text:
        return text
    
    # For titles (first heading), preserve the full text including colons
    if is_title:
        # Only limit extremely long titles to avoid capturing paragraphs
        if len(text) > 200:
            # Find a good break point (space, comma, period)
            for i in range(180, min(200, len(text))):
                if text[i] in ' ,.':
                    text = text[:i].strip()
                    break
            else:
                text = text[:180].strip()
        return text
    
    # For non-title headings, split at colon and take only the first part
    if ':' in text:
        text = text.split(':', 1)[0].strip()
    
    # Limit heading length to reasonable size (avoid capturing paragraph text)
    # Most headings should be under 100 characters
    if len(text) > 100:
        # Find a good break point (space, comma, period)
        for i in range(80, min(100, len(text))):
            if text[i] in ' ,.':
                text = text[:i].strip()
                break
        else:
            text = text[:80].strip()
    
    return text


def extract_headings(markdown_text):
    headings = []

    # Pattern 1: Markdown headers (##, ###, etc.)
    markdown_pattern = r'^(#{1,6})\s*(.*)$'

    # Pattern 2: Numbered sections (1., 1.1, 1.1.1, etc.)
    numbered_pattern = r'^(\d+(?:\.\d+)*\.?)\s+(.*)$'

    # Pattern 3: Parenthetical numbers (1), 2), etc.)
    paren_pattern = r'^(\d+\))\s+(.*)$'

    lines = markdown_text.split('\n')
    is_first_heading = True

    for line in lines:
        line = line.strip()

        # Check for markdown headers
        md_match = re.match(markdown_pattern, line)
        if md_match:
            level = len(md_match.group(1))  # Count # symbols
            heading_text = clean_heading_text(md_match.group(2).strip(), is_first_heading)
            headings.append({
                'type': 'markdown',
                'level': level,
                'number': None,
                'text': heading_text,
                'raw': line
            })
            is_first_heading = False
            continue

        # Check for numbered sections
        num_match = re.match(numbered_pattern, line)
        if num_match:
            number = num_match.group(1)
            heading_text = clean_heading_text(num_match.group(2).strip(), is_first_heading)
            level = number.count('.') + 1
            headings.append({
                'type': 'numbered',
                'level': level,
                'number': number,
                'text': heading_text,
                'raw': line
            })
            is_first_heading = False
            continue

        # Check for parenthetical numbers
        paren_match = re.match(paren_pattern, line)
        if paren_match:
            number = paren_match.group(1)
            heading_text = clean_heading_text(paren_match.group(2).strip(), is_first_heading)
            headings.append({
                'type': 'parenthetical',
                'level': 1,
                'number': number,
                'text': heading_text,
                'raw': line
            })
            is_first_heading = False

    return headings


# Usage
source = "https://arxiv.org/pdf/2312.07305v1"
converter = DocumentConverter()
result = converter.convert(source)
markdown = result.document.export_to_markdown()

headings = extract_headings(markdown)

# Display results
for h in headings:
    indent = "  " * (h['level'] - 1)
    if h['number']:
        print(f"{indent}{h['number']} {h['text']}")
    else:
        print(f"{indent}{'#' * h['level']} {h['text']}")