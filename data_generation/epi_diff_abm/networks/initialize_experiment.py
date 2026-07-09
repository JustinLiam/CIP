'''Import libraries'''
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import pickle
import shutil

from gen_mob_nw import generate_mobility_networks
from custom_population import customize


state_abbrev = os.environ.get("EPI_STATE_ABBREV", "AL")
target_counties = [
    '01045',
]

if os.environ.get("EPI_COUNTIES"):
    target_counties = [
        county.strip().zfill(5)
        for county in os.environ["EPI_COUNTIES"].replace(",", " ").split()
        if county.strip()
    ]


state_dict = {
    'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8, 'CT': 9, 'DE': 10,
    'DC': 11, 'FL': 12, 'GA': 13, 'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18,
    'IA': 19, 'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24, 'MA': 25,
    'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30, 'NE': 31, 'NV': 32,
    'NH': 33, 'NJ': 34, 'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39,
    'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45, 'SD': 46, 'TN': 47,
    'TX': 48, 'UT': 49, 'VT': 50, 'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55,
    'WY': 56
}



def save_dict_to_file(dictionary, filename="agent_ages_data.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dictionary, f)

def load_dict_from_file(filename="agent_ages_data.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}


county_to_ageslist_dict = load_dict_from_file()

# Ask user if they want to run the loop
user_input = input("Customize population? (yes/no): ").strip().lower()

if user_input != "yes" and user_input != "no":
    print("Error: Invalid input for customize population")
    exit(1)

if user_input == "yes":
    print("-----------------------------------------------------------------")
    print("Customizing population!")
    print("-----------------------------------------------------------------")
    print()

    # Reset ages list dictionary for new population
    county_to_ageslist_dict = {}

    #Data Directory
    county_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'data/state_data/{state_abbrev}')

    #Results Directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    results_path_dir = 'population_data/' + state_abbrev + '_population_data'
    results_dir = os.path.join(data_dir, results_path_dir)
    # Ensure the results_dir directory exists
    os.makedirs(results_dir, exist_ok=True)

    #Randomly Generate Data Directory
    rand_gen_path_dir = 'population_data/' + state_abbrev + '_rand_gen_stats_dir'
    rand_gen_dir = os.path.join(data_dir, rand_gen_path_dir)
    os.makedirs(rand_gen_dir, exist_ok=True)

    fips_pattern = re.compile(r"^\d{5}$")

    # Iterate through all folders in `data_dir` with a progress bar
    for folder_name in tqdm(os.listdir(os.path.join(county_data_dir)), desc="Processing Counties", unit="folder"):
        folder_path = os.path.join(county_data_dir, folder_name)
        # Check if it's a directory and the name matches the FIPS pattern
        if os.path.isdir(folder_path) and fips_pattern.match(folder_name):
            county = str(folder_name)
            # Check if the state code matches the first two digits of the FIPS code
            state_code = int(county[:2])  # Extract first two digits and convert to integer
            if state_code == state_dict[state_abbrev]:
                # Customize population for county
                print(f'Customizing population for county {folder_name}...')
                ages_list = customize(data_dir=county_data_dir, results_dir=results_dir, rand_gen_dir=rand_gen_dir, county=county)
                assert county not in county_to_ageslist_dict, f"Error: '{county}' is already a key in the dictionary!"
                county_to_ageslist_dict[county] = ages_list

    save_dict_to_file(county_to_ageslist_dict)  # Save it for future use

elif not county_to_ageslist_dict:
    assert user_input == "no"
    print("Error: No existing data found. Please customize the population at least once.")
    exit(1)

else:
    assert user_input == "no"
    print("Using previously saved population data.")

experiment_type = input("Run full experiment (lighthouse) or a test (local)? (full/test): ").strip().lower()

if experiment_type != "full" and experiment_type != "test":
    print("Error: Invalid input for experiment type")
    exit(1)



if experiment_type == "full":

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'networks')
    output_dir = os.path.join(data_dir, "covid_output_causal")

else:
    assert experiment_type == "test"

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'networks')
    output_dir = os.path.join(data_dir, "test_output")

base_dir = output_dir

if not os.path.exists(base_dir):
    os.makedirs(base_dir)



print("Generating mobility networks!")

num_steps = int(input("Enter number of timesteps: "))

print(f"Number of timesteps: {num_steps}")

if num_steps <= 0 or num_steps > 366:
    print("Error: Number of timesteps out of bounds")
    exit(1)
county = None

to_pickle_string = input("Use pickle files for mobility networks instead of csv? (yes/no): ").strip().lower()

if to_pickle_string != "yes" and to_pickle_string != "no":
    print("Error: Invalid input for to pickle")
    exit(1)

to_pickle = to_pickle_string == "yes"


for fips_code in tqdm(target_counties, desc="Processing Counties", unit="county"):
    county_path = os.path.join(base_dir, fips_code)
    os.makedirs(county_path, exist_ok=True)

    memory_path = os.path.join(county_path, "memory.txt")
    # Only create the file if it doesn't exist
    if not os.path.exists(memory_path):
        open(memory_path, "w").close()
    print("-----------------------------------------------------------------")
    generate_mobility_networks(state_abbrev=state_abbrev, county=fips_code, output_dir=county_path, num_steps=num_steps, ages_list=county_to_ageslist_dict[fips_code], to_pickle=to_pickle)
    print("-----------------------------------------------------------------")

print()
print("Initialization of Experiment Finished!")
