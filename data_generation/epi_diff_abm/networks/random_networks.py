import random
import os
import numpy as np
import networkx as nx
from sim_gen_utils import custom_watts_strogatz_graph
import pandas as pd
from scipy.stats import nbinom
import pickle

def get_num_random_interactions(age, random_network_params_dict,
                                child_upper_ix, adult_upper_ix):
    '''Returns the number of random interactions for an agent based on the age
        Takes the mean and sd from the random_network_params_dict
    '''
    if age <= child_upper_ix:
        mean = random_network_params_dict['CHILD']['mu']
        sd = random_network_params_dict['CHILD']['sigma']
    elif age <= adult_upper_ix:
        mean = random_network_params_dict['ADULT']['mu']
        sd = random_network_params_dict['ADULT']['sigma']
    else:
        mean = random_network_params_dict['ELDERLY']['mu']
        sd = random_network_params_dict['ELDERLY']['sigma']


    mean = max(mean, 0.1)
    sd = max(sd, (mean + 0.1)**0.5)

    p = mean / (sd * sd)
    n = mean * mean / (sd * sd - mean)

    n = max(n, 0.1)
    p = min(max(p, 0.01), 0.99)

    num_interactions = nbinom.rvs(n, p)
    return num_interactions

def create_and_write_random_networks(num_agents, agents_ages, num_steps,
                                     random_nw_infile, child_upper_ix,
                                     adult_upper_ix, county, google_param_changes,
                                     to_pickle, output_dir):
    '''Creates and writes the random networks to a file'''
    if not os.path.isfile(random_nw_infile):
        print(
            'The file with random network parameters not found at location {}'.
            format(random_nw_infile))
        raise FileNotFoundError

    # Load the network parameters
    random_nw_df = pd.read_csv(random_nw_infile, index_col=0)
    random_network_params_dict = {
        a: {
            'mu': random_nw_df.loc[a, 'mu'],
            'sigma': random_nw_df.loc[a, 'sigma']  # Fixed a typo ('mu' → 'sigma')
        }
        for a in random_nw_df.index.to_list()
    }

    outfile_path = os.path.join(output_dir, "randnets")
    os.makedirs(outfile_path, exist_ok=True)

    # Iterate over timesteps with a progress bar
    for t in range(num_steps):
        # Adjust mu for this timestep
        adjustment_factor = 1.0 + (google_param_changes[t] / 100.0)

        for group in random_network_params_dict:
            random_network_params_dict[group]['mu'] *= adjustment_factor
            random_network_params_dict[group]['mu'] = max(random_network_params_dict[group]['mu'], 0.1)
            random_network_params_dict[group]['mu'] = min(random_network_params_dict[group]['mu'], 10)

        # Generate random interactions based on updated `mu`
        agents_random_interactions = [
            get_num_random_interactions(age, random_network_params_dict,
                                        child_upper_ix, adult_upper_ix)
            for age in agents_ages
        ]

        interactions_list = []
        for agent_id in range(num_agents):
            interactions_list.extend([agent_id] *
                                     agents_random_interactions[agent_id])
        random.shuffle(interactions_list)
        edges_list = [(interactions_list[i], interactions_list[i + 1])
                      for i in range(len(interactions_list) - 1)]
        G = nx.Graph()
        G.add_edges_from(edges_list)

        if to_pickle:
            outfile = os.path.join(outfile_path, f"{t}.pkl")
            # Save to pickle
            with open(outfile, "wb") as f:
                pickle.dump(G, f)
        else:
            outfile = os.path.join(outfile_path, f"{t}.csv")
            nx.write_edgelist(G, outfile, delimiter=',', data=False)
