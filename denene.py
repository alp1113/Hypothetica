import urllib.request as libreq

with libreq.urlopen('http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=10') as url:
    r = url.read()
print(r)
