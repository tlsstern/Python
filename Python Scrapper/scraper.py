from googleapiclient.discovery import build
import json

# pip install --upgrade google-api-python-client json
# Search Engine ID from     https://programmablesearchengine.google.com/controlpanel/all
# API KEY FROM              https://developers.google.com/custom-search/v1/introduction

API_KEY = 'API_KEY'
SEARCH_ENGINE_ID = 'SEARCH_ENGINE_ID'

def google_search_api(search_term, num_results=10):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        kwargs = {'q': search_term, 'cx': SEARCH_ENGINE_ID, 'num': num_results}
        result = service.cse().list(**kwargs).execute()
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    'title': item['title'],
                    'link': item['link'],
                    'snippet': item['snippet']
                })
        return search_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_results(results, filename="google_results.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to '{filename}'.")

if __name__ == "__main__":
    search_term = input("What would you like to search for? ")

    while True:
        try:
            num_results_str = input("How many results would you like (1-10)? ")
            num_results = int(num_results_str)
            if 1 <= num_results <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    search_results = google_search_api(search_term, num_results)

    if search_results:
        save_results(search_results)
    else:
        print("No results found using the Google Custom Search API.")