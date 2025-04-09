from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup

# pip install --upgrade selenium beautifulsoup4 time

CHROMEDRIVER_PATH = "chromedriver-win64\chromedriver.exe"

service = Service(CHROMEDRIVER_PATH)
options = webdriver.ChromeOptions()
options.add_argument("--window-size=1920,1080")

driver = None
print("Drive initialized")

try:
    driver = webdriver.Chrome(service=service, options=options)

    search_url = "https://noseryoung.ch/"
    file = "index.html"
    driver.get(search_url)

    time.sleep(2)

    page_html = driver.page_source

    with open(file, "a", encoding="utf-8") as f:
        f.write(page_html)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if driver:
        print("driver was successfully initialized")
        driver.quit()