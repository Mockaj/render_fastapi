import requests
from rag.config.settings import settings

def enhance_query_with_langtail(query: str) -> str:
    url = f"https://api.langtail.com/{settings.LANGTAIL_WORKSPACE}/{settings.LANGTAIL_PROJECT}/{settings.LANGTAIL_PROMPT}/{settings.LANGTAIL_ENVIRONMENT}"
    headers = {
        "X-API-Key": settings.LANGTAIL_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "variables": {"query": query},
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Extract the assistant's reply
        enhanced_query = data['choices'][0]['message']['content']
        return enhanced_query
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to enhance query with Langtail: {e}")
    except KeyError:
        raise ValueError("Unexpected response format from Langtail")