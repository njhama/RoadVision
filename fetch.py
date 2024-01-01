import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
from csv import writer
import logging

def get_args():
    parser = argparse.ArgumentParser(description="Download Google Street View images based on GPS coordinates.")
    parser.add_argument("--cities", required=True, type=str, help="Folder with JSON files containing addresses per city.")
    parser.add_argument("--output", default='images/', type=str, help="Output folder for images (default: images/).")
    parser.add_argument("--icount", default=25000, type=int, help="Number of images to pull (default: 25000).")
    parser.add_argument("--key", required=True, type=str, help="Google Street View API Key.")
    return parser.parse_args()

def load_cities(args):
    cities = []
    for city in os.listdir(args.cities):
        with open(os.path.join(args.cities, city)) as f:
            coordinates = [json.loads(line)['geometry']['coordinates'] for line in f]
            cities.append(coordinates)
            print(f'Loaded {len(coordinates)} addresses from {city}')
    return cities

def download_images(args, cities):
    os.makedirs(args.output, exist_ok=True)
    coord_output_file = open(os.path.join(args.output, 'picture_coords.csv'), 'w', newline='')
    csv_writer = writer(coord_output_file)
    
    for i in tqdm(range(args.icount), desc="Downloading images"):
        city_index = randint(0, len(cities) - 1)
        city = cities[city_index]

        if not city:
            logging.warning(f"No more coordinates left for city index: {city_index}")
            continue

        addressLoc = city.pop(randint(0, len(city) - 1))
        params = {
            'key': args.key,
            'size': '640x640',
            'location': f"{addressLoc[1]},{addressLoc[0]}",
            'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
            'pitch': '20',
            'fov': '90'
        }

        try:
            response = requests.get('https://maps.googleapis.com/maps/api/streetview', params)
            response.raise_for_status()
            with open(os.path.join(args.output, f'street_view_{i}.jpg'), "wb") as file:
                file.write(response.content)
            csv_writer.writerow([addressLoc[1], addressLoc[0]])
        except requests.RequestException as e:
            logging.error(f"Error downloading image {i}: {e}")

    coord_output_file.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    cities = load_cities(args)
    download_images(args, cities)
    logging.info("Image downloading process completed.")

if __name__ == '__main__':
    main()
