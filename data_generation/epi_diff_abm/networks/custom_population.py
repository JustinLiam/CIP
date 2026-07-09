import os
import pandas as pd
import json
import numpy as np
import count_people as cntppl
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t59", "60t69", "70t79", "80A"],  # Age ranges for adults
    "children_list": ["0t9", "10t19"],  # Age range for children
}

MOBILITY_MAPPING = json.load(open('mobility_mapping.json'))

def customize(data_dir, results_dir, rand_gen_dir, county, num_agents = None):
    #My implementation

    #------------------------MANAGING DATA PATHS------------------------
    # Check if the directory exists
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"The directory '{results_dir}' does not exist. Please create it first.")

    # Check if data_dir exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please ensure it exists and is accessible.")

    #County data path
    county_data_dir = os.path.join(data_dir, county)
    if not os.path.isdir(county_data_dir):
        raise FileNotFoundError(f"The file '{county_data_dir}' does not exist. Please ensure it is available in '{data_dir}'.")

    #Set data directories:
    age_groups_data_path = os.path.join(county_data_dir, 'agents_ages.csv')
    household_sizes_data_path = os.path.join(county_data_dir, 'agents_household_sizes.csv')
    occupations_data_path = os.path.join(county_data_dir, 'agents_occupations.csv')

    # Check if each file path exists
    if not os.path.isfile(age_groups_data_path):
        raise FileNotFoundError(f"The file '{age_groups_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")

    if not os.path.isfile(household_sizes_data_path):
        raise FileNotFoundError(f"The file '{household_sizes_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")

    if not os.path.isfile(occupations_data_path):
        raise FileNotFoundError(f"The file '{occupations_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")


    #Load age group data
    AGE_GROUPS_DATA = pd.read_csv(age_groups_data_path, encoding='ISO-8859-1')
    # Load household data
    HOUSEHOLD_DATA = pd.read_csv(household_sizes_data_path, encoding='ISO-8859-1')
    #load occupation data
    OCCUPATION_DATA = pd.read_csv(occupations_data_path, encoding='ISO-8859-1')


    #Create list of ages, household, and occupation stats
    ages_list = []
    household_list = []
    occupations_list = []

    #Create list of random ages in order of age groups
    for index, row in AGE_GROUPS_DATA.iterrows():
        lower = -1
        upper = -1
        age_group = row['Age']
        match = re.findall(r'\d+', age_group)
        # Convert extracted strings to integers
        if len(match) == 1:
            lower = int(match[0])
            assert lower == 80
            upper = 100
        elif len(match) == 2:  # Ensure there are two numbers found
            lower, upper = int(match[0]), int(match[1])

        assert lower >= 0 and lower <= 80 and upper >= 9 and upper <= 100

        num_age_group = int(row['Number'])
        for i in range(num_age_group):
            random_int = np.random.randint(lower, upper + 1)
            ages_list.append(random_int)

    #Create list of household sizes in order
    for index, row in HOUSEHOLD_DATA.iterrows():
        houseSize = row['Household Size']
        num_household_size = int(row['Number'])
        houseSum = houseSize * num_household_size
        for i in range(houseSum):
            household_list.append(houseSize)


    #Create list of occupations in order
    total_jobs = 0
    for index, row in OCCUPATION_DATA.iterrows():
        total_jobs += int(row['Number'])

    occ_fractions = {}
    for index, row in OCCUPATION_DATA.iterrows():
        fraction = int(row['Number']) / total_jobs
        occupation = str(row['Occupation'])
        occ_fractions[occupation] = fraction

    #Randomly shuffle the lists to randomize population:
    np.random.shuffle(ages_list)
    np.random.shuffle(household_list)

    # print(f'Size of Ages List: {len(ages_list)}')
    # print(f'Size of Household List: {len(household_list)}')
    # print(f'Size of Occupations List: {len(occupations_list)}')
    # quit()


    #SAVE RANDOMLY GENERATED DATA
    # Define the output file path
    output_file_path_str = county + '_population_data.txt'
    output_file_path = os.path.join(rand_gen_dir, output_file_path_str)

    # Write each list to a separate line in the text file
    with open(output_file_path, 'w') as file:
        file.write("Ages: " + ", ".join(map(str, ages_list)) + "\n")
        file.write("Household Sizes: " + ", ".join(map(str, household_list)) + "\n")
        file.write("Occupations: " + ", ".join(occupations_list) + "\n")

    print(f"County data summary saved to {output_file_path}")


    # quit()

    #------------------------EVENING OUT POPULATION SIZES OF DATASETS------------------------
    largest_pop = cntppl.largest_total_population(data_path=county_data_dir)

    # print(largest_pop)

    if largest_pop == "Occupation":
        print("Invalid data, occupation data larger than age/household data!")
        exit(1)

    if largest_pop == "Ages":
        diff = len(ages_list) - len(household_list)
        assert diff < len(ages_list)
        ages_list = ages_list[:-diff]

    elif largest_pop == "Household":
        diff = len(household_list) - len(ages_list)
        assert diff < len(household_list)
        household_list = household_list[:-diff]

    else:
        assert largest_pop == "Equal"

    assert len(ages_list) == len(household_list)
    pop_size = len(ages_list)

    print(f'Population size = {pop_size}')

    # quit()

    #------------------------ASSIGNING HOUSEHOLD IDS------------------------
    household_ids = [0] * len(household_list)
    houseSizes = np.arange(1, 7)
    household_id = 0
    for houseSize in houseSizes:
        houseCounter = 0
        household_id += 1   # To assert that if numPplInHouseholdSize % HouseholdSize != 0, the incorrect household_id doesn't carry over (we can allow jumps in household_id as long as they're distinct)
        for i in range(len(household_list)):
            if household_list[i] == houseSize:
                household_ids[i] = household_id
                houseCounter += 1
                if houseCounter == houseSize:
                    houseCounter = 0
                    household_id += 1

    assert all(x != 0 for x in household_ids)

    #------------------------CREATING FINAL DATAFRAME------------------------
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=["ID", "Age", "Household Size", "Occupations"])

    # Generate data as lists
    ids = range(0, pop_size)
    occupations = [None] * pop_size

    # Create DataFrame using the lists directly
    df = pd.DataFrame({
        "ID": ids,
        "Age": ages_list,
        "Household Size": household_list,
        "Occupations": occupations,
        "Household ID": household_ids
    })

    # Count people with age between 18 and 65
    eligible_count = df[(df["Age"] >= 18) & (df["Age"] <= 65)].shape[0]

    #Create occupation list
    unemployement_rate = 0.07
    num_employed = int(eligible_count * (1 - unemployement_rate))
    for occ, frac in occ_fractions.items():
        num_of_occ = int(frac * num_employed)
        for i in range(num_of_occ):
            occupations_list.append(occ)

    # Check if there are enough eligible people for the occupations
    assert len(occupations_list) < eligible_count

    # Assign occupations to eligible people
    eligible_indices = df[(df["Age"] >= 18) & (df["Age"] <= 65)].index.tolist()

    # Shuffle the eligible indices to ensure random assignment
    np.random.shuffle(eligible_indices)

    # Assign occupations
    for i, occupation in zip(eligible_indices, occupations_list):
        df.loc[i, "Occupations"] = occupation

    # Assign "" to remaining eligible people without an occupation
    for i in eligible_indices[len(occupations_list):]:
        df.loc[i, "Occupations"] = "Unemployed"


    #Convert dataframe to csv and store in results_dir
    # Define the path for the CSV file
    file_path = os.path.join(results_dir, f"{county}_population.csv")

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)  # Set index=False to avoid saving row indices

    print(f'Population data saved for county {county} to {file_path}')
    print()

    # Return age list since needed for random network creation in experiment initialization
    return ages_list
