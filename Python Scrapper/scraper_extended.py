from googleapiclient.discovery import build
import json
import csv
import sqlite3

# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# Search Engine ID from     https://programmablesearchengine.google.com/controlpanel/all
# API KEY FROM              https://developers.google.com/custom-search/v1/introduction

API_KEY = 'API_KEY'
SEARCH_ENGINE_ID = 'SEARCH_ENGINE_ID'

def google_search_api(search_term, num_results=10, country=None, language=None, dateRestrict=None, fileType=None, **kwargs):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        search_params = {'q': search_term, 'cx': SEARCH_ENGINE_ID, 'num': num_results}

        if country:
            search_params['gl'] = country
        if language:
            search_params['lr'] = language
        if dateRestrict:
            search_params['dateRestrict'] = dateRestrict
        if fileType:
            search_params['fileType'] = fileType

        search_params.update(kwargs)

        result = service.cse().list(**search_params).execute()
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

def save_results(results, filename="google_results", output_format="json", db_name="search_results.db"):
    if output_format == "json":
        with open(filename + ".json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to '{filename}.json'")

    elif output_format == "csv":
        with open(filename + ".csv", 'w', encoding='utf-8', newline='', errors='replace') as f:
            writer = csv.writer(f)
            if results:
                writer.writerow(results[0].keys())
                for row in results:
                    writer.writerow(row.values())
        print(f"Results saved to '{filename}.csv'")

    elif output_format == "db":
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {filename}")
            if results:
                columns = ", ".join(results[0].keys())
                cursor.execute(f"CREATE TABLE {filename} ({columns})")
                placeholders = ", ".join(["?"] * len(results[0]))
                insert_query = f"INSERT INTO {filename} VALUES ({placeholders})"
                for row in results:
                    cursor.execute(insert_query, list(row.values()))
            conn.commit()
            conn.close()
            print(f"Results saved to table '{filename}' in '{db_name}'")
        except sqlite3.Error as e:
            print(f"Error saving to database: {e}")

    else:
        print(f"Invalid output format: {output_format}")

def get_user_input(prompt, allowed_values=None):
    while True:
        user_input = input(prompt).strip().lower()
        if allowed_values is None or user_input in allowed_values:
            return user_input
        else:
            print(f"Invalid input. Please enter one of: {', '.join(allowed_values)}")

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

    extra_params = {}

    country = get_user_input("Enter country: us, uk, ch, de: ")
    if country:
        extra_params['country'] = country
    language = get_user_input("Enter language: lang_en, lang_de: ")
    if language:
        extra_params['language'] = language
    dateRestrict = get_user_input("Enter dateRestriction: d[number], w[number], m[number], y[number]: ")
    if dateRestrict:
        extra_params['dateRestrict'] = dateRestrict
    fileType = get_user_input("Enter fileType: pdf, doc, docx, xls, xlsx, ppt, pptx: ")
    if fileType:
        extra_params['fileType'] = fileType

    output_format = get_user_input("Enter output format json, csv, db: ", allowed_values=["json", "csv", "db"])

    search_results = google_search_api(search_term, num_results, **extra_params)

    if search_results:
        save_results(search_results, output_format=output_format)
    else:
        print("No results found (crazy)")