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

    for line_idx, line in enumerate(lines):
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
                'raw': line,
                'line_index': line_idx
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
                'raw': line,
                'line_index': line_idx
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
                'raw': line,
                'line_index': line_idx
            })
            is_first_heading = False

    return headings


def get_text_between_headings(markdown_text, heading_name):
    """
    Extract text from a specified heading to the next heading.
    
    Args:
        markdown_text: The markdown text from the document
        heading_name: The name of the heading to search for (case-insensitive, partial match)
    
    Returns:
        A dictionary containing:
        - 'found': Boolean indicating if the heading was found
        - 'heading': The matched heading text
        - 'content': The text between the heading and the next heading
        - 'next_heading': The next heading (if exists)
    """
    headings = extract_headings(markdown_text)
    lines = markdown_text.split('\n')
    
    # Find the heading that matches the search term (case-insensitive)
    heading_name_lower = heading_name.lower()
    matched_heading = None
    matched_index = None
    
    for idx, heading in enumerate(headings):
        # Check if the heading text contains the search term
        heading_text = heading['text'].lower()
        heading_raw = heading['raw'].lower()
        
        if heading_name_lower in heading_text or heading_name_lower in heading_raw:
            matched_heading = heading
            matched_index = idx
            break
    
    if not matched_heading:
        return {
            'found': False,
            'heading': None,
            'content': None,
            'next_heading': None,
            'available_headings': [h['text'] for h in headings]
        }
    
    # Get the start line (line after the matched heading)
    start_line = matched_heading['line_index'] + 1
    
    # Get the end line (line before the next heading, if exists)
    if matched_index + 1 < len(headings):
        next_heading = headings[matched_index + 1]
        end_line = next_heading['line_index']
    else:
        next_heading = None
        end_line = len(lines)
    
    # Extract the content between the headings
    content_lines = lines[start_line:end_line]
    content = '\n'.join(content_lines).strip()
    
    return {
        'found': True,
        'heading': matched_heading['text'],
        'content': content,
        'next_heading': next_heading['text'] if next_heading else None
    }


if __name__ == "__main__":
    # Usage example
    source = "https://arxiv.org/pdf/2312.07305v1"
    converter = DocumentConverter()
    result = converter.convert(source)
    markdown = result.document.export_to_markdown()
    print(markdown)
    # Extract all headings
    headings = extract_headings(markdown)

    # Display all headings
    print("=" * 80)
    print("ALL HEADINGS IN DOCUMENT:")
    print("=" * 80)
    for h in headings:
        indent = "  " * (h['level'] - 1)
        if h['number']:
            print(f"{indent}{h['number']} {h['text']}")
        else:
            print(f"{indent}{'#' * h['level']} {h['text']}")

    print("\n" + "=" * 80)
    print("EXTRACT TEXT FROM A SPECIFIC HEADING:")
    print("=" * 80)

    # Example: Get text from a specific heading
    # You can change this to any heading name you want
    heading_to_search = input("\nEnter the heading name you want to extract (or press Enter to skip): ").strip()

    if heading_to_search:
        result = get_text_between_headings(markdown, heading_to_search)

        if result['found']:
            print(f"\n✓ Found heading: {result['heading']}")
            print(f"Next heading: {result['next_heading'] if result['next_heading'] else 'End of document'}")
            print("\n" + "-" * 80)
            print("CONTENT:")
            print("-" * 80)
            print(result['content'])
        else:
            print(f"\n✗ Heading '{heading_to_search}' not found.")
            print("\nAvailable headings:")
            for h in result['available_headings']:
                print(f"  - {h}")