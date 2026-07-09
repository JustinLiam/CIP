import pandas as pd
import numpy as np
import os

def get_dataframes_dict(county_code):
    county_fips = int(county_code)
    # Note: These paths are likely expected to be in a specific location
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datapath_template = "data/full_google_mob_data/{}_US_Region_Mobility_Report.csv"
    datapath_template = os.path.join(root_dir, datapath_template)

    final_dict = {}
    for year in [2020, 2021, 2022]:
        df = pd.read_csv(datapath_template.format(year))
        df = df[df["census_fips_code"] == county_fips]
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        final_dict[year] = df

    return final_dict
