import pandas as pd
from requests import get
from utils import chunks
import threading as th
import numpy as np
import csv

def create_dataset(array, n_thread):
  print(f'Thread {n_thread} started')
  for i, item in enumerate(array):
    try:
      url, price = item
      print(f'Thread {n_thread}: Notebook {i+1}/{len(array)}')
      response = get(url)
      notebook_data = response.json()
      data = {}

      data['name'] = notebook_data['name']
      data['url'] = notebook_data['url']
      data['id'] = notebook_data['id']
      data['slug'] = notebook_data['slug']
      data['picture_url'] = notebook_data['picture_url']
      data['price'] = int(price.replace('$', '').replace('.', ''))

      specs = notebook_data['specs']
      # Specs
      data['score_general'] = specs['score_general']
      data['score_games'] = specs['score_games']
      data['score_mobility'] = specs['score_mobility']
      data['weight'] = specs['weight']
      data['power_adapter_power'] = specs['power_adapter_power']

      data['operating_system_family_name'] = specs['operating_system_family_name']
      
      data['screen_size_value'] = specs['screen_size_size']
      data['screen_resolution_unicode'] = specs['screen_resolution_unicode']
      data['screen_refresh_rate_value'] = specs['screen_refresh_rate_value']
      data['battery_mwh'] = specs['battery_mwh']
      data['default_bucket'] = specs['default_bucket']

      # Procesador
      data['processor_unicode'] = specs['processor_unicode']
      data['processor_frequency_value'] = specs['processor_frequency_value']
      data['processor_thread_count_name'] = specs['processor_thread_count_name']
      data['processor_p_core_count'] = specs['processor_p_core_count']
      data['processor_thread_count_value'] = specs['processor_thread_count_value']
      data['processor_tdp'] = specs['processor_tdp']
      data['processor_speed_score'] = specs['processor_speed_score']

      # Gpu
      data['main_gpu_name'] = specs['main_gpu']['unicode']
      data['main_gpu_speed_score'] = specs['main_gpu']['speed_score']

      # Ram
      data['ram_quantity_value'] = specs['ram_quantity_value']
      data['ram_frequency_value'] = specs['ram_frequency_value']
      data['ram_type_name'] = specs['ram_type_name']

      # Largest storage drive
      data['sd_capacity_value'] = specs['largest_storage_drive']['capacity_value']
      data['sd_rpm_value'] = specs['largest_storage_drive']['rpm_value']
      data['sd_drive_type_name'] = specs['largest_storage_drive']['drive_type_name']

      # Counters
      data['storage_drives_count'] = len(specs['storage_drive'])
      data['gpus_count'] = len(specs['gpus'])

      notebook_data_list.append(data)
    except Exception as e:
      print(f'Error in thread {n_thread}, in notebook {i+1}/{len(array)}: {e}\n {url}')
  
  print(f'Thread {n_thread} finished')

if __name__ == '__main__':
  notebook_data_list = []
  
  notebok_url_dataset = pd.read_csv('notebook-dataset/notebook_url_array.tsv', sep='\t')
  notebook_url_list = notebok_url_dataset[['public_api_url', 'price']].to_numpy()

  with_threading = True
  if with_threading:
    threads = []
    for i, chunk in enumerate(chunks(notebook_url_list, 40)):
      thread = th.Thread(target=create_dataset, args=(chunk,i,))
      threads.append(thread)
      thread.start()
    for thread in threads:
      thread.join()
  else:
    create_dataset(notebook_url_list)

  notebook_data_numpy = np.array(notebook_data_list)
  print(notebook_data_numpy[0].keys())

  with open('notebook-dataset/notebook_data.tsv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=notebook_data_numpy[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(notebook_data_list)