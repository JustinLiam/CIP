import pickle
import pandas as pd
import os
import json
import requests
from io import StringIO

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

def process_fips_data(fips_code):
    # Define paths
    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_daily_path = os.path.join(project_root, "data", "delphi_county_data", f"{fips_code}_data.csv")
    output_dir = os.path.join(project_root, "data", "processed_data", fips_code, "202010-202104")
    os.makedirs(output_dir, exist_ok=True)

    # Define final date range to save
    final_start = pd.to_datetime("2020-10-26")
    final_end   = pd.to_datetime("2021-04-11")

    # Helper: replace a negative value with the average of nearest non-negative neighbors
    def fix_negative(series):
        series = series.copy()
        for i in range(len(series)):
            if series[i] < 0:
                # find previous non-negative
                prev_idx = i - 1
                while prev_idx >= 0 and series[prev_idx] < 0:
                    prev_idx -= 1
                # find next non-negative
                next_idx = i + 1
                while next_idx < len(series) and series[next_idx] < 0:
                    next_idx += 1
                # assign replacement
                if prev_idx >= 0 and next_idx < len(series):
                    series[i] = (series[prev_idx] + series[next_idx]) / 2
                elif prev_idx >= 0:
                    series[i] = series[prev_idx]
                elif next_idx < len(series):
                    series[i] = series[next_idx]
                else:
                    series[i] = 0
        return series

    # Process daily
    if os.path.exists(input_daily_path):
        # Load all of 2020 and 2021
        df = pd.read_csv(input_daily_path, parse_dates=["time_value"])
        df = df[df["time_value"].dt.year.isin([2020, 2021])]
        df = df[["time_value", "cases", "deaths"]].rename(columns={"time_value": "date"})

        # Fix negative values
        df["cases"]  = fix_negative(df["cases"])
        df["deaths"] = fix_negative(df["deaths"])

        # Aggregate weekly on full data
        df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly_df = df.groupby("week", as_index=False)["deaths"].sum()
        weekly_df = weekly_df.rename(columns={"week": "date"})

        # Slice to the final desired range before saving
        df_final    = df[(df["date"] >= final_start) & (df["date"] <= final_end)]
        weekly_final = weekly_df[(weekly_df["date"] >= final_start) & (weekly_df["date"] <= final_end)]

        # Save daily
        daily_output_path = os.path.join(output_dir, "daily_data.csv")
        df_final.to_csv(daily_output_path, index=False)
        print(f"Saved daily data to {daily_output_path}")

        # Save weekly
        weekly_output_path = os.path.join(output_dir, "weekly_data.csv")
        weekly_final.to_csv(weekly_output_path, index=False)
        print(f"Saved weekly data to {weekly_output_path}")

    else:
        print(f"Daily data file not found for FIPS {fips_code}.")


def create_county_folder(state_abbrev, fips_code):
    # Construct the county directory name using state abbreviation + FIPS
    county_dir_name = f"pop{fips_code}"
    county_dir = os.path.join('./populations', county_dir_name)
    os.makedirs(county_dir, exist_ok=True)

    # Create mapping.json
    mapping_data = {
        "age": [
            "20t29",
            "30t39",
            "40t49",
            "50t64",
            "65A",
            "U19"
        ]
    }
    with open(os.path.join(county_dir, 'mapping.json'), 'w') as f:
        json.dump(mapping_data, f, indent=2)


    mobility_dir = os.path.join(county_dir, "mobility_networks")
    os.makedirs(mobility_dir, exist_ok=True)
    zero_file = os.path.join(mobility_dir, "0.csv")
    pd.DataFrame([[0, 0]]).to_csv(zero_file, index=False, header=False)

    # Build population file path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    pop_file = f"population_data/{state_abbrev}_population_data/{fips_code}_population.csv"
    pop_file = os.path.join(data_dir, pop_file)

    try:
        df = pd.read_csv(pop_file)

        # Create age.csv
        df['Age'].to_csv(os.path.join(county_dir, 'age.csv'), index=False, header=['age'])

        # Create age.pickle
        age_mapping = []
        for age in df['Age']:
            if age < 20:
                mapped = 5  # U19
            elif 20 <= age < 30:
                mapped = 0  # 20t29
            elif 30 <= age < 40:
                mapped = 1  # 30t39
            elif 40 <= age < 50:
                mapped = 2  # 40t49
            elif 50 <= age < 65:
                mapped = 3  # 50t64
            else:
                mapped = 4  # 65A
            age_mapping.append(mapped)

        # Create pandas Series with name 'age'
        age_series = pd.Series(age_mapping, name='age')
        with open(os.path.join(county_dir, 'age.pickle'), 'wb') as f:
            pickle.dump(age_series, f)

        # Create disease_stages.csv
        stages = [0.0] * len(df)
        pd.DataFrame(stages, columns=['stages']).to_csv(
            os.path.join(county_dir, 'disease_stages.csv'),
            index=False
        )

        print(f"Successfully created files for {county_dir_name}")

    except FileNotFoundError:
        print(f"Population file not found: {pop_file}")
    except Exception as e:
        print(f"Error processing {county_dir_name}: {str(e)}")

    # Generate the intervention file

    start_date = pd.to_datetime("2020-10-26")
    dates = pd.date_range(start_date, periods=182, freq='D')
    interventions_df = pd.DataFrame({"date": dates})

    policy = pd.read_csv("data/202436_w_policy2.csv")
    policy["end_date"] = pd.to_datetime(policy["end_date"])
    state_policy = policy[policy["region"] == state_abbrev].copy()

    # ensure sorted by date
    state_policy.sort_values("end_date", inplace=True)
    state_policy.reset_index(drop=True, inplace=True)

    def map_intervention(x):
        return 0 if x in [0,1] else 1

    # For each daily date, find the weekly row whose end_date >= date
    school_vals = []
    occ_vals = []

    for d in interventions_df["date"]:
        row = state_policy[state_policy["end_date"] >= d]
        if len(row) == 0:
            school_vals.append(0)
            occ_vals.append(0)
        else:
            row = row.iloc[0]
            school_vals.append(map_intervention(row["C1_School closing"]))
            occ_vals.append(map_intervention(row["C2_Workplace closing"]))

    interventions_df["school_intervention"] = school_vals
    interventions_df["occ_intervention"] = occ_vals

    # convert date to string for output
    interventions_df["date"] = interventions_df["date"].dt.strftime("%Y-%m-%d")

    interventions_df["t"] = range(len(interventions_df))

    # save intervention.csv inside county_dir
    interventions_df.to_csv(
        os.path.join(county_dir, "intervention.csv"),
        index=False
    )

    print(f"Created intervention.csv for {county_dir_name}")


def generate_all_county_data(state_abbrv, county):
    process_fips_data(county)
    create_county_folder(state_abbrv, county)


for county in target_counties:
    generate_all_county_data(state, county)
