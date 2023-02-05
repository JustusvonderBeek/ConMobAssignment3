import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json, scipy

from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def exportToPdf(fig, filename):
    """
    Exports the current plot to file. Both in the 10:6 and 8:6 format (for thesis and slides.
    """

    # Saving to 8:6 format (no name change)
    fig.set_figwidth(8)
    fig.set_figheight(6)

    fig.savefig(filename, bbox_inches='tight', format='pdf')
    print(f"Wrote output to '{filename}'")

# --------------------------------------------------------------------------
# DATA HANDLING
# --------------------------------------------------------------------------

def extractPingLatencies(args):
    """
    Expecting the input JSON file location. This file should contain RIPE Atlas Ping Measurements
    Extracting the min,max,average latencies
    Returing the result as pandas data frame.
    """

    input_file = args.input[0]

    with open(input_file, "r") as file:
        measurements = json.load(file)
    
    skip_counter = 0
    latency_dict = defaultdict(list)
    for measure in tqdm(measurements):
        if measure["min"] == -1 or measure["max"] == -1 or measure["avg"] == -1:
            skip_counter += 1
            continue
        latency_dict["min"].append(measure["min"])
        latency_dict["max"].append(measure["max"])
        latency_dict["avg"].append(measure["avg"])

    data = pd.DataFrame(latency_dict)
    # print(data.to_markdown())
    print(f"Skipped '{skip_counter}' nodes because of missing PING results!")
    return data

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------

def plotPingLatencyCDF(args):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency CDF')

    data = extractPingLatencies(args)

    fig, ax = plt.subplots()

    sns.kdeplot(data=data["avg"], cumulative=True, label="Avg. RTT")
    sns.kdeplot(data=data["min"], cumulative=True, label="Min. RTT")
    sns.kdeplot(data=data["max"], cumulative=True, label="Max. RTT")

    plt.legend(title="RTT type", loc="upper left")
    plt.xlabel('RTT [ms]')
    plt.ylabel('CDF')
    plt.title("CDF of Ping RTT")
    plt.grid("both")

    exportToPdf(fig, args.output)


# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate CDF plots to compare the ping latency')
    parser.add_argument('-i','--input', action="append", default=[], required=True, help="The path to the JSON file containg the latency information")
    parser.add_argument('-o','--output', type=str, default="pingRTT.pdf", help="The file in which the resulting plot is stored")

    args = parser.parse_args()

    plotPingLatencyCDF(args)