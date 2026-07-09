import pandas as pd
from collections import Counter
import networkx as nx
import numpy as np
import os
import pickle
from sim_gen_utils import custom_watts_strogatz_graph, normal_watts_strogatz_graph
# from isolate_google_mob_data import get_google_mob_data
from isolate_county_data import get_dataframes_dict
from random_networks import create_and_write_random_networks
from tqdm import tqdm


def generate_mobility_networks(state_abbrev, county, output_dir, num_steps, ages_list, to_pickle):
    print(f"County {county}:")

    mobility_dir = os.path.join(output_dir, "mobility_networks")
    os.makedirs(mobility_dir, exist_ok=True)

    # Load individual data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    agent_data_path = f"population_data/{state_abbrev}_population_data/{county}_population.csv"
    agent_data_path_dir = os.path.join(data_dir, agent_data_path)
    individuals = pd.read_csv(agent_data_path_dir)
    num_agents = individuals.shape[0]

    # Load network parameters
    occ_params_data_path = "network_parameters/occupation_network_parameters.csv"
    occupation_params = pd.read_csv(occ_params_data_path, index_col="Occupation")

    random_params_data_path = "network_parameters/random_network_parameters.csv"
    random_params = pd.read_csv(random_params_data_path, index_col="Age")

    # Map occupation names to indices
    occupation_names = occupation_params.index.to_list()
    occupation_to_ix = {name: idx for idx, name in enumerate(occupation_names)}

    # Filter individuals with valid occupations
    individuals['OccupationID'] = individuals['Occupations'].map(occupation_to_ix)
    individuals_with_occupations = individuals.dropna(subset=['OccupationID'])

    # Group individuals by occupation
    occupation_groups = {
        occ: individuals_with_occupations[individuals_with_occupations['OccupationID'] == occ]['ID'].tolist()
        for occ in occupation_to_ix.values()
    }

    # Count population per occupation
    occupation_population = Counter(individuals_with_occupations['OccupationID'])

    # Group individuals by household
    households = individuals.groupby('Household ID').groups

    print("Generating Occupation, School, and Household Networks:")

    output_occ_dir = f"{mobility_dir}/occnets"
    os.makedirs(output_occ_dir, exist_ok=True)

    output_school_dir = f"{mobility_dir}/schoolnets"
    os.makedirs(output_school_dir, exist_ok=True)

    # Retrieve MI county Google dataframes:
    google_MI_df_dict = get_dataframes_dict(county)

    start_date_str = "2020-10-12"
    year = 2020
    curr_df = google_MI_df_dict[year]

    if start_date_str not in curr_df["date"].values:
        raise ValueError(f"Start date {start_date_str} not found in 2021 mobility data.")

    start_index = curr_df.index[curr_df["date"] == start_date_str][0]
    print(f"Starting from {start_date_str} (index {start_index})")


    children = individuals[(individuals['Age'] >= 0) & (individuals['Age'] <= 19)]
    agents_school = children['ID'].to_list()
    mu_school = random_params.loc['CHILD', 'mu']
    sigma_school = random_params.loc['CHILD', 'sigma']

    index = start_index
    isOutOfData = False

    # Make a list of floats to keep track of random network parameters that will influence mu
    google_parameters_randomnets = []

    # Outer loop for time steps
    for t in range(num_steps):
        # Check if at the end of a yearly dataframe
        is_last_row = index == curr_df.index[-1] + 1
        if is_last_row:
            if year == 2022:
                print("No more timesteps left, end of available mobility data.")
                exit(0)
            else:
                year += 1
                curr_df = google_MI_df_dict[year]
                index = 0


        # This timestep's date:
        todays_date = curr_df.at[index, "date"]

        # Random network google parameters extraction:
        retail_rec = curr_df.at[index, "retail_and_recreation_percent_change_from_baseline"]
        if np.isnan(retail_rec):
            retail_rec = 0.0

        groc_pharm = curr_df.at[index, "grocery_and_pharmacy_percent_change_from_baseline"]
        if np.isnan(groc_pharm):
            groc_pharm = 0.0

        parks = curr_df.at[index, "parks_percent_change_from_baseline"]
        if np.isnan(parks):
            parks = 0.0

        transit = curr_df.at[index, "transit_stations_percent_change_from_baseline"]
        if np.isnan(transit):
            transit = 0.0

        random_net_param_change = (retail_rec + groc_pharm + parks + transit) / 4.0
        google_parameters_randomnets.append(random_net_param_change)

        #Occupation data google adjustment (mu adjustment):
        occ_mu_perc_change = curr_df.at[index, "workplaces_percent_change_from_baseline"]
        if np.isnan(occ_mu_perc_change):
            occ_mu_perc_change = 0.0
        occ_decimal = occ_mu_perc_change / 100.0
        occ_factor = 1.0 + occ_decimal

        index += 1

        G_all = nx.Graph()

        # Inner loop for occupation groups with tqdm
        for occ, agents in occupation_groups.items():
            n_agents = len(agents)
            if n_agents > 1:  # Avoid empty or trivial networks
                mu = occupation_params.loc[occupation_names[occ], 'mu']
                mu *= occ_factor
                rewire = occupation_params.loc[occupation_names[occ], 'rewire']
                avg_degree = min(int(np.round(mu)), n_agents - 1)  # Ensure avg_degree < n_agents

                # Generate network using custom implementation
                G_occ = custom_watts_strogatz_graph(
                    n=n_agents,                      # Number of nodes
                    k=avg_degree,                    # Average degree
                    p=[rewire, agents]               # Rewiring probability and node names
                )

                # No relabeling needed
                # # Map agent indices back to original IDs
                # id_mapping = {i: agents[i] for i in range(n_agents)}
                # G_occ = nx.relabel_nodes(G_occ, id_mapping)

                # Merge this occupation subgraph into the timestep graph
                G_all.add_nodes_from(G_occ.nodes(data=True))
                G_all.add_edges_from(G_occ.edges())

        # Save network to file
        if(to_pickle):
            outfile = os.path.join(output_occ_dir, f"{t}.pkl")
            # Save to pickle
            with open(outfile, "wb") as f:
                pickle.dump(G_all, f)
        else:
            outfile = os.path.join(output_occ_dir, f"{t}.csv")
            nx.write_edgelist(G_all, outfile, delimiter=",", data=False)

        # School network
        n_agents = len(individuals)
        G_school = normal_watts_strogatz_graph(
            n=n_agents,             # Number of nodes
            agents=agents_school,          # Agent IDs
            mu=mu_school,                  # Degree average
            sigma=sigma_school             # Degree standard deviation
        )

        # Save network to file
        if(to_pickle):
            outfile = os.path.join(output_school_dir, f"{t}.pkl")
            # Save to pickle
            with open(outfile, "wb") as f:
                pickle.dump(G_school, f)
        else:
            outfile = os.path.join(output_school_dir, f"{t}.csv")
            nx.write_edgelist(G_school, outfile, delimiter=",", data=False)

    # Generation Household networks
    n_agents = len(individuals)
    if n_agents > 1:  # Avoid empty or trivial networks
        # Generate network using custom implementation
        G = custom_watts_strogatz_graph(
            n=n_agents,             # Number of nodes
            k=0,                    # Average degree = 0
            p=[0, agents]           # Rewiring probability = 0 and node names
        )

        # Add edges for agents in the same household with progress bar
        for household_id, members in households.items():
            # Convert dataframe indices to actual agent IDs
            household_members = individuals.loc[members, 'ID'].tolist()

            # Add edges between all pairs of household members
            if len(household_members) > 1:  # Skip single-member households
                for i in range(len(household_members)):
                    for j in range(i + 1, len(household_members)):
                        G.add_edge(household_members[i], household_members[j])

        # Save network to file
        if(to_pickle):
            outfile = os.path.join(mobility_dir, f"HOUSEHOLD_NETWORK.pkl")
            # Save to pickle
            with open(outfile, "wb") as f:
                pickle.dump(G, f)
        else:
            outfile = os.path.join(mobility_dir, f"HOUSEHOLD_NETWORK.csv")
            nx.write_edgelist(G, outfile, delimiter=",", data=False)

    print("Generating Random Network:")
    random_nw_params_path = "network_parameters/random_network_parameters.csv"
    create_and_write_random_networks(num_agents=num_agents, agents_ages=ages_list, num_steps=num_steps, random_nw_infile=random_nw_params_path,
                                     child_upper_ix=19, adult_upper_ix=65, county=county, google_param_changes=google_parameters_randomnets,
                                     to_pickle=to_pickle, output_dir=mobility_dir)