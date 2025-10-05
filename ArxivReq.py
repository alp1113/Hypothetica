import urllib.request
import urllib.parse
from datetime import datetime, timedelta


def search_arxiv(terms=None, operator=None, category=None, search_in="all",
                 max_results=10, start=0, sort_by=None, sort_order="descending",
                 date_from=None, date_to=None):
    """
    Search arXiv API for papers.

    Args:
        terms: Single term (str) or list of terms, e.g., "RAG" or ["RAG", "agents"]
        operator: "AND" or "OR" - how to combine multiple terms
        category: arXiv category to filter by, e.g., "cs.AI"
        search_in: Where to search - "all", "ti" (title), "abs" (abstract), "au" (author)
        max_results: Number of results to return (default: 10)
        start: Starting index for pagination (default: 0)
        sort_by: "relevance", "lastUpdatedDate", "submittedDate"
        sort_order: "ascending" or "descending" (default: "descending")
        date_from: Start date for filtering - datetime object or "YYYYMMDDHHMM" string
        date_to: End date for filtering - datetime object or "YYYYMMDDHHMM" string

    Returns:
        String containing the Atom XML response

    Examples:
        # Papers from last 6 months
        search_arxiv("RAG", date_from=datetime.now() - timedelta(days=180))

        # Papers from 2024
        search_arxiv("agents", date_from="202401010000", date_to="202412312359")

        # Recent AI papers from last year
        search_arxiv(category="cs.AI", date_from=datetime.now() - timedelta(days=365))
    """

    query_parts = []

    # Build search query from terms
    if terms:
        if isinstance(terms, str):
            terms = [terms]

        if len(terms) == 1:
            query_parts.append(f"{search_in}:{terms[0]}")
        elif operator:
            query_parts.append(f" {operator} ".join([f"{search_in}:{term}" for term in terms]))
        else:
            query_parts.append(" ".join([f"{search_in}:{term}" for term in terms]))

    # Add category filter if specified
    if category:
        query_parts.append(f"cat:{category}")

    # Add date range filter if specified
    if date_from or date_to:
        # Convert datetime objects to string format if needed
        if isinstance(date_from, datetime):
            date_from = date_from.strftime("%Y%m%d%H%M")
        if isinstance(date_to, datetime):
            date_to = date_to.strftime("%Y%m%d%H%M")

        # Set defaults if only one date is provided
        if not date_from:
            date_from = "200001010000"  # arXiv started in 2000
        if not date_to:
            date_to = datetime.now().strftime("%Y%m%d%H%M")

        query_parts.append(f"submittedDate:[{date_from} TO {date_to}]")

    # Combine all parts
    if len(query_parts) > 1:
        query = " AND ".join([f"({part})" for part in query_parts])
    elif len(query_parts) == 1:
        query = query_parts[0]
    else:
        raise ValueError("Must provide either terms, category, or date range")

    # Build the full URL with parameters
    params = {
        'search_query': query,
        'start': start,
        'max_results': max_results
    }

    # Add sorting parameters if specified
    if sort_by:
        params['sortBy'] = sort_by
        params['sortOrder'] = sort_order

    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

    # Make the request
    response = urllib.request.urlopen(url)
    return response.read().decode('utf-8')


# Helper functions for common date ranges
def last_days(days):
    """Get datetime for N days ago"""
    return datetime.now() - timedelta(days=days)


def last_months(months):
    """Get approximate datetime for N months ago (30 days per month)"""
    return datetime.now() - timedelta(days=months * 30)


def search_multiple_topics(topics, **kwargs):
    """
    Search multiple topics separately with individual API calls.

    Args:
        topics: List of topics to search separately
        **kwargs: Additional parameters passed to search_arxiv

    Returns:
        Dictionary with results for each topic
    """
    results = {}
    for topic in topics:
        print(f"Searching for: {topic}...")
        results[topic] = search_arxiv(topic, **kwargs)
        # Be nice to the API - add a delay between requests
        import time
        time.sleep(3)  # 3 second delay as recommended

    return results


# Usage:

# Example usage:
if __name__ == "__main__":
    # Papers from last 6 months
    # print("RAG papers from last 6 months:")
    # result = search_arxiv("RAG", date_from=last_months(6), max_results=10)
    # print(result[:500])

    # print("\n" + "=" * 50 + "\n")

    # Papers from last year
    # print("AI agent papers from last year:")
    # raglist=['electrons','RAG','IOT']
    # result = search_arxiv(raglist, date_from=last_days(365), max_results=10,operator="OR")
    # print(result)
    topics = ["agents", "RAG", "IOT"]
    all_results = search_multiple_topics(topics, date_from=last_days(365), max_results=10)

    # Access individual results
    print("RAG papers:", all_results["RAG"])
    print("Agent papers:", all_results["agents"])
    print("IOT papers:", all_results["IOT"])

    # print("\n" + "=" * 50 + "\n")

    # # Papers from 2024 only
    # print("All RAG papers from 2024:")
    # result = search_arxiv("RAG", date_from="202401010000", date_to="202412312359", max_results=10)
    # print(result[:500])

    # print("\n" + "=" * 50 + "\n")

    # # Last 30 days, newest first
    # print("Recent papers from last 30 days:")
    # result = search_arxiv(["RAG", "retrieval"],
    #                       date_from=last_days(30),
    #                       sort_by="submittedDate",
    #                       sort_order="descending",
    #                       max_results=5)
    # print(result[:500])