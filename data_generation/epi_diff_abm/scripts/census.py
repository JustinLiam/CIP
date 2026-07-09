"""From census API, obtain multiple distributions for a county
    - household size distribution
    - age distribution
    - occupation distribution
    Variables as per https://api.census.gov/data/2021/acs/acs5/variables.html
"""

import requests
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

state = os.environ.get("EPI_STATE_ABBREV", "AL")
target_counties = [
    '01045',
]

if os.environ.get("EPI_COUNTIES"):
    target_counties = [
        county.strip().zfill(5)
        for county in os.environ["EPI_COUNTIES"].replace(",", " ").split()
        if county.strip()
    ]


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _census_get(url, county_fips, label):
    timeout_seconds = _env_int("EPI_CENSUS_TIMEOUT_SECONDS", 60)
    retries = max(1, _env_int("EPI_CENSUS_RETRIES", 5))
    retry_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code == 200 or response.status_code not in retry_statuses:
                return response
            print(
                f"Census request returned status={response.status_code} "
                f"for county={county_fips} label={label} attempt={attempt}/{retries}",
                flush=True,
            )
        except requests.RequestException as exc:
            print(
                f"Census request failed for county={county_fips} label={label} "
                f"attempt={attempt}/{retries}: {type(exc).__name__}",
                flush=True,
            )
            if attempt >= retries:
                raise RuntimeError(
                    f"Census request failed for county={county_fips} label={label} "
                    f"after {retries} attempts"
                ) from None
        if attempt < retries:
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(
        f"Census request failed for county={county_fips} label={label} after {retries} attempts"
    )


def obtain_household_size_distribution(state, county_fips, census_api_key):
    """ Obtain household size distribution for county from Census data"""

    # Define the variables you want to query
    household_variables = [
        "B11016_001E", "B11016_003E", "B11016_011E", "B11016_004E",
        "B11016_012E", "B11016_005E", "B11016_013E", "B11016_006E",
        "B11016_014E", "B11016_007E", "B11016_008E", "B11016_015E",
        "B11016_016E"
    ]

    # Define the dictionary with the column mapping
    household_size_acs_var_map = {
        "1":
        "B11016_001E - B11016_003E - B11016_011E - B11016_004E -"
        "B11016_012E - B11016_005E - B11016_013E - B11016_006E -"
        "B11016_014E - B11016_007E - B11016_008E - B11016_015E -"
        "B11016_016E",
        "2":
        "B11016_003E + B11016_011E",
        "3":
        "B11016_004E + B11016_012E",
        "4":
        "B11016_005E + B11016_013E",
        "5":
        "B11016_006E + B11016_014E",
        "6":
        "B11016_007E + B11016_008E + B11016_015E + B11016_016E",
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(household_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = _census_get(url, county_fips, "household")

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        print(f"Obtaining Household Data")
        for new_column, formula in household_size_acs_var_map.items():
            # print(f"{new_column}/6")
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(household_size_acs_var_map.keys())]
    """ Save with format as csv file
        Household Size,Number
        1,329114
        2,306979
        3,139176
        4,115757
        5,45162
        6,30375
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Household Size", 0: "Number"})
    output_path = f"./data/state_data/{state}/{county_fips}/"
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}agents_household_sizes.csv", index=False)


def obtain_age_distribution(state, county_fips, census_api_key):
    """ Obtain age distribution for county from Census data """

    age_variables = [
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",
        "B01001_007E", "B01001_027E", "B01001_028E", "B01001_029E",
        "B01001_030E", "B01001_031E", "B01001_008E", "B01001_009E",
        "B01001_010E", "B01001_011E", "B01001_012E", "B01001_032E",
        "B01001_033E", "B01001_034E", "B01001_035E", "B01001_036E",
        "B01001_013E", "B01001_014E", "B01001_015E", "B01001_016E",
        "B01001_017E", "B01001_037E", "B01001_038E", "B01001_039E",
        "B01001_040E", "B01001_041E", "B01001_018E", "B01001_019E",
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
        "B01001_024E", "B01001_025E", "B01001_042E", "B01001_043E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E",
        "B01001_048E", "B01001_049E"
    ]

    # Define the dictionary with the column mapping
    age_acs_var_map = {
        "AGE_0_9":
        "B01001_003E + B01001_004E + B01001_027E + B01001_028E",
        "AGE_10_19":
        "B01001_005E + B01001_006E + B01001_007E + B01001_029E +"
        "B01001_030E + B01001_031E",
        "AGE_20_29":
        "B01001_008E + B01001_009E + B01001_010E + "
        "B01001_011E + B01001_032E + B01001_033E + "
        "B01001_034E + B01001_035E",
        "AGE_30_39":
        "B01001_012E + B01001_013E + B01001_036E + B01001_037E",
        "AGE_40_49":
        "B01001_014E + B01001_015E + B01001_038E + B01001_039E",
        "AGE_50_59":
        "B01001_016E + B01001_017E + B01001_040E + B01001_041E",
        "AGE_60_69":
        "B01001_018E + B01001_019E + B01001_020E + B01001_021E +"
        "B01001_042E + B01001_043E + B01001_044E + B01001_045E",
        "AGE_70_79":
        "B01001_022E + B01001_023E + B01001_046E + B01001_047E",
        "AGE_80":
        "B01001_024E + B01001_025E + B01001_048E + B01001_049E"
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(age_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = _census_get(url, county_fips, "age")

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        # print('-------------------')
        print(f"Obtaining Age Data")
        for new_column, formula in age_acs_var_map.items():
            # print(f"{new_column}/6")
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(age_acs_var_map.keys())]
    """ Save with format as csv file
        Age,Number
        AGE_0_9,278073
        AGE_10_19,258328
        AGE_20_29,317005
        AGE_30_39,359688
        AGE_40_49,323457
        AGE_50_59,307938
        AGE_60_69,229274
        AGE_70_79,109487
        AGE_80,69534
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Age", 0: "Number"})
    output_path = f"./data/state_data/{state}/{county_fips}/"
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}agents_ages.csv", index=False)


def obtain_occupation_distribution(state, county_fips, census_api_key):
    """ Obtain occupation distribution for county from Census data
        Using NAICS sectors as Abueg et al. 2021, npj Digital Medicine

    """
    # Define the dictionary with the sector code mapping
    inverted_naics_sectors = {
        'AGRICULTURE': ['11'],
        'MINING': ['21'],
        'UTILITIES': ['22'],
        'CONSTRUCTION': ['23'],
        'MANUFACTURING': ['31', '32', '33'],
        'WHOLESALETRADE': ['42'],
        'RETAILTRADE': ['44', '45'],
        'TRANSPORTATION': ['48', '49'],
        'INFORMATION': ['51'],
        'FINANCEINSURANCE': ['52'],
        'REALESTATERENTAL': ['53'],
        'SCIENTIFICTECHNICAL': ['54'],
        'ENTERPRISEMANAGEMENT': ['55'],
        'WASTEMANAGEMENT': ['56'],
        'EDUCATION': ['61'],
        'HEALTHCARE': ['62'],
        'ART': ['71'],
        'FOOD': ['72'],
        'OTHER': ['81', '92'],  # adding public administration to other
    }

    df = [['Occupation', 'Number']]
    # convert example api.census.gov/data/2017/ecnbasic?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=us:*&NAICS2017=54&key=YOUR_KEY_GOES_HERE
    base_url = "https://api.census.gov/data/2017/ecnbasic"
    # print('-------------------')
    print(f"Obtaining Occupation Data")
    for naics_sector in inverted_naics_sectors:
        # print(naics_sector)
        # print('-------------------')
        for naics_sector_code in inverted_naics_sectors[naics_sector]:
            # print(naics_sector, naics_sector_code)
            url = f"{base_url}?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&NAICS2017={naics_sector_code}&key={census_api_key}"
            # Make the API request
            response = _census_get(url, county_fips, f"occupation_{naics_sector_code}")
            # Check the response status code
            if response.status_code == 200:
                # The request was successful, so parse the JSON response
                results = json.loads(response.content)
                row = results[1][:2]
                # label sector
                row[0] = naics_sector
                df.append(row)
            else:
                # The request failed, so print the error message
                # print(response.status_code, response.reason)
                row = [naics_sector, 0]
                df.append(row)

    # construct dataframe
    df = pd.DataFrame(df[1:], columns=df[0])
    # convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    # aggregate by sector but do not sort by sector
    df = df.groupby(['Occupation'], sort=False).sum()
    # reset index
    df = df.reset_index()
    output_path = f"./data/state_data/{state}/{county_fips}/"
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}agents_occupations.csv", index=False)


if __name__ == "__main__":
    # Define your Census API key
    CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

    for county_fips in target_counties:
        # create directory if it does not exist
        if not os.path.exists(f"./data/state_data/{state}/{county_fips}"):
            os.makedirs(f"./data/state_data/{state}/{county_fips}")
        print(f'Creating data for {county_fips}')
        obtain_household_size_distribution(state, county_fips, CENSUS_API_KEY)
        obtain_age_distribution(state, county_fips, CENSUS_API_KEY)
        obtain_occupation_distribution(state, county_fips, CENSUS_API_KEY)
        print()
