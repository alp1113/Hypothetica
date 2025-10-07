import re
from docling.document_converter import DocumentConverter


def clean_heading_text(text, is_title=False):
    """Clean heading text by removing content after colons and limiting length."""
    if not text:
        return text

    if is_title:
        if len(text) > 200:
            for i in range(180, min(200, len(text))):
                if text[i] in ' ,.':
                    text = text[:i].strip()
                    break
            else:
                text = text[:180].strip()
        return text

    if ':' in text:
        text = text.split(':', 1)[0].strip()

    if len(text) > 100:
        for i in range(80, min(100, len(text))):
            if text[i] in ' ,.':
                text = text[:i].strip()
                break
        else:
            text = text[:80].strip()

    return text


def extract_headings(markdown_text):
    headings = []
    markdown_pattern = r'^(#{1,6})\s*(.*)$'
    numbered_pattern = r'^(\d+(?:\.\d+)*\.?)\s+(.*)$'
    paren_pattern = r'^(\d+\))\s+(.*)$'

    lines = markdown_text.split('\n')
    is_first_heading = True

    for line_num, line in enumerate(lines):
        line_stripped = line.strip()

        md_match = re.match(markdown_pattern, line_stripped)
        if md_match:
            level = len(md_match.group(1))
            heading_text = clean_heading_text(md_match.group(2).strip(), is_first_heading)
            headings.append({
                'type': 'markdown',
                'level': level,
                'number': None,
                'text': heading_text,
                'raw': line_stripped,
                'line_num': line_num
            })
            is_first_heading = False
            continue

        num_match = re.match(numbered_pattern, line_stripped)
        if num_match:
            number = num_match.group(1)
            heading_text = clean_heading_text(num_match.group(2).strip(), is_first_heading)
            level = number.count('.') + 1
            headings.append({
                'type': 'numbered',
                'level': level,
                'number': number,
                'text': heading_text,
                'raw': line_stripped,
                'line_num': line_num
            })
            is_first_heading = False
            continue

        paren_match = re.match(paren_pattern, line_stripped)
        if paren_match:
            number = paren_match.group(1)
            heading_text = clean_heading_text(paren_match.group(2).strip(), is_first_heading)
            headings.append({
                'type': 'parenthetical',
                'level': 1,
                'number': number,
                'text': heading_text,
                'raw': line_stripped,
                'line_num': line_num
            })
            is_first_heading = False

    return headings


def get_text_between_headings(markdown_text, start_heading, end_heading=None):
    """
    Extract text between two headings.

    Args:
        markdown_text: The full markdown text
        start_heading: Text of the starting heading (partial match)
        end_heading: Text of the ending heading (partial match). If None, gets to end of document

    Returns:
        String containing the text between the headings
    """
    lines = markdown_text.split('\n')
    headings = extract_headings(markdown_text)

    # Find the start heading
    start_idx = None
    for h in headings:
        if start_heading.lower() in h['text'].lower():
            start_idx = h['line_num']
            break

    if start_idx is None:
        return f"Start heading '{start_heading}' not found"

    # Find the end heading
    end_idx = len(lines)
    if end_heading:
        for h in headings:
            if h['line_num'] > start_idx and end_heading.lower() in h['text'].lower():
                end_idx = h['line_num']
                break

    # Extract text between headings (excluding the headings themselves)
    content_lines = lines[start_idx + 1:end_idx]
    return '\n'.join(content_lines).strip()


# Usage
source = "https://arxiv.org/pdf/2312.07305v1"
converter = DocumentConverter()
result = converter.convert(source)
markdown = result.document.export_to_markdown()

headings = extract_headings(markdown)

# Display all headings
print("=== ALL HEADINGS ===")
for h in headings:
    indent = "  " * (h['level'] - 1)
    if h['number']:
        print(f"{indent}{h['number']} {h['text']}")
    else:
        print(f"{indent}{'#' * h['level']} {h['text']}")

# Example: Extract text between two headings
print("\n=== TEXT BETWEEN HEADINGS ===")
text = get_text_between_headings(markdown, "Introduction", "References")
print(f"Text between 'Introduction' and 'Related Work':")
print(text)