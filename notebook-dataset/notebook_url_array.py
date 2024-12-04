from typing import List
from bs4 import BeautifulSoup
from requests import get
import csv
import threading as th

from utils import chunks

TOTAL_PAGES = 80
BASE_URL = 'https://www.solotodo.cl'
PUBLIC_API_URL = 'https://publicapi.solotodo.com'
DATA_KEYS = ['public_api_url', 'notebook_url', 'price']
def webscraper(list_pages: List[int]):
    for i in list_pages:
        print(f'Página {i}/{TOTAL_PAGES}')
        response = get(f'{BASE_URL}/notebooks', params={'page': i})
        html_response = BeautifulSoup(response.text, 'html.parser')
        notebooks_container = html_response.select_one('div.css-w19e2l')

        for notebook in notebooks_container.select('div.css-1wxaqej'):
            id = notebook.select_one('a').get('href').split('/')[-1].split('-')[0]
            public_api_url = f'{PUBLIC_API_URL}/products/{id}'
            notebook_url_array.append({
                "public_api_url": public_api_url,
                "notebook_url": BASE_URL + notebook.select_one('a').get('href'),
                "price": notebook.select_one('div.css-1lytq14').text
            })
            # notebook_url_array.append(notebook_data)

if __name__ == '__main__':
    notebook_url_array: List[dict] = []
    with_threading = True
    if with_threading:
        threads = []
        for chunk in chunks(range(1, TOTAL_PAGES + 1), 5):
            threads.append(th.Thread(target=webscraper, args=(chunk,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        webscraper(range(1, TOTAL_PAGES + 1))

    with open('notebook_url_array.tsv', 'w', encoding='utf-8', newline='') as data_source:
        writer = csv.DictWriter(data_source, fieldnames=DATA_KEYS, delimiter='\t')
        writer.writeheader()
        writer.writerows(notebook_url_array)