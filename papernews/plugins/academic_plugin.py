import pluggy
import requests
from typing import List
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")

@hookimpl
def fetch_documents(config: dict) -> List[RawDocument]:
    """Fetches academic open-access metadata via the Unpaywall API."""
    documents = []
    
    # Assumes your sources.toml has a block like:
    # [academic]
    # dois = ["10.1038/nature12373", "10.1126/science.1245"]
    academic_config = config.get("academic", {})
    dois = academic_config.get("dois", [])

    # You must append an email to use the free Unpaywall API
    email = "papernews_user@example.com" 

    for doi in dois:
        print(f"Querying Unpaywall for DOI: {doi}")
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", "Unknown Title")
                
                # We pull the publisher and OA status as our "raw text" for the LLM to format
                publisher = data.get("publisher", "Unknown Publisher")
                is_oa = data.get("is_oa", False)
                oa_url = data.get("best_oa_location", {}).get("url_for_pdf", "No PDF available")
                
                raw_text = (
                    f"Title: {title}\n"
                    f"Publisher: {publisher}\n"
                    f"Open Access: {is_oa}\n"
                    f"PDF Link: {oa_url}\n"
                    f"Summary: A new study published by {publisher} regarding {title}."
                )
                
                doc = RawDocument(
                    source_id=doi,
                    content_type="academic_pdf",
                    raw_text=raw_text,
                    metadata={"title": title, "url": oa_url}
                )
                documents.append(doc)
            else:
                print(f"Unpaywall returned {resp.status_code} for {doi}")
        except Exception as e:
            print(f"Failed to fetch DOI {doi}: {e}")
            
    return documents