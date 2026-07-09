import os
import pandas as pd


def largest_total_population(data_path):
    total_age_pop = 0
    total_house_pop = 0
    total_occ_pop = 0

    agents_ages_csv = os.path.join(data_path, "agents_ages.csv")
    agents_household_sizes_csv = os.path.join(data_path, "agents_household_sizes.csv")
    agents_occupations_csv = os.path.join(data_path, "agents_occupations.csv")

    df_ages = pd.read_csv(agents_ages_csv)
    df_house = pd.read_csv(agents_household_sizes_csv)
    df_occ = pd.read_csv(agents_occupations_csv)

    for index, row in df_ages.iterrows():
        number = row['Number']
        total_age_pop += number

    for index, row in df_house.iterrows():
        houseSize = row['Household Size']
        number = row['Number']
        total_house_pop += houseSize * number

    for index, row in df_occ.iterrows():
        number = row['Number']
        total_occ_pop += number

    if (total_occ_pop > total_age_pop) or (total_occ_pop > total_house_pop):
        return "Occupation"

    if total_age_pop == total_house_pop:
        return "Equal"

    if total_age_pop > total_house_pop:
        return "Ages"

    else:
        assert total_house_pop > total_age_pop
        return "Household"