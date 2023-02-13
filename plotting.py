import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json, scipy, ast
import geopandas as gpd

from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from shapely.geometry import Point
from geopandas import GeoDataFrame

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

    # Check if folder exists and create the path if failed
    dirname, fname = os.path.split(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig.savefig(filename, bbox_inches='tight', format='pdf')
    print(f"Wrote output to '{filename}'")

# --------------------------------------------------------------------------
# DATA HANDLING
# --------------------------------------------------------------------------

def extractPingLatencies(args):
    """
    Expecting the input CSV file location. This file should contain RIPE Atlas Ping Measurements
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

def extractGeolocation(inputs, location_file):
    """
    Expecting the CSV file(s) with IDs. Extracting all IDs and combine them into single list.
    Returning a list of Points of Geolocations.
    """

    if type(inputs) != list:
        inputs = [inputs]

    ids = []
    for file in inputs:
        print(f"Reading in '{file}'...")
        data = pd.read_csv(file)
        for index, row in data.iterrows():
            id_list = ast.literal_eval(row["IDs"])
            # print(id_list)
            ids.extend(id_list)

    # print(ids)
    locations = list()
    with open(location_file, "r") as location_input:
        probes = json.load(location_input)
        probes = probes["objects"]
        for id in ids:
            location = next(((node["longitude"],node["latitude"]) for node in probes if node["id"] == id), "Unknown")
            locations.append(Point(location))

    # print(locations)
    return locations

def convertTimestampToTime(timestamp):
    """
    Expecting a timestamp as input
    Returning the converted timestamp as HH:MM:SS
    """
    return time.strftime("%H:%M", time.gmtime(timestamp))

def convertTimestampToWeekday(timestamp):
    """
    Expecting a timestamp as input
    Returning the converted timestamp as Weekday
    """
    return time.strftime("%a", time.gmtime(timestamp))

def filterTimeOfDay(dataframe, start, end):
    """
    Expecting the PING measurement data and the time interval which should be filtered.
    Returning the filtered DataFrame where PINGs happened only within the interval.
    """

    start = datetime.time(int(start))
    end = datetime.time(int(end))
    print(f"{start}")
    dataframe["Time"] = dataframe["Timestamp"].apply(convertTimestampToTime)
    # print(f"{dataframe['Time']}")
    dataframe = dataframe.loc[dataframe["Time"] ]
    exit(1)

    return dataframe

def filterInvalid(dataframe):
    """
    This method can be used to filter invalid PINGs from the DataFrame.
    """

    # Currently we only filter pings that did not reached the destination
    return dataframe.loc[dataframe["Avg"] > 0]

def filterAccessTechnology(dataframe, technology):
    """
    Expecting the PING measurement data and the access technology.
    Returning the filtered DataFrame where PINGs used the given technology.
    """

    return dataframe.loc[dataframe["Technology"] == technology]

def filterIntraContinent(dataframe, continent):
    """
    Expecting the PING measurement data and the continent as input.
    Returning the filtered DataFrame where connections only go from CONTINENT -> CONTINENT.
    """

    dataframe = dataframe.loc[dataframe["Continent"] == continent]
    return data.loc[dataframe["Datacenter Continent"] == continent]

def filterInterContinent(dataframe, source, dest):
    """
    Expecting the PING measurement data and the source and destination continent as input.
    Returning the filtered DataFrame where connections go from SOURCE -> DEST.
    """

    dataframe = dataframe.loc[dataframe["Continent"] == source]
    return dataframe.loc[dataframe["Datacenter Continent"] == dest]

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------

def plotPingLatencyCDF(args):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency CDF')

    # data = extractPingLatencies(args)
    data = pd.read_csv(args.input[0], na_filter=False)

    fig, ax = plt.subplots()

    # test_data = data.loc[data["Continent"] == "ME"]
    test_data = data.loc[data["Technology"] == "WIFI"]
    # test_data = test_data.loc[data["Technology"] == "WIFI"]
    test_data = test_data.loc[data["Avg"] > 0]
    microsoft = test_data.loc[data["Datacenter Company"] == "MICROSOFT"]
    # microsoft["Timestamp"] = pd.to_datetime(microsoft["Timestamp"], format="%Y%m%d")
    # print(microsoft.to_markdown())
    # print(f"{len(test_data)}")
    # print(f"{microsoft.iloc[0]['Timestamp']}")
    print(f"Time conversion {convertTimestampToTime(microsoft.iloc[0]['Timestamp'])}")
    filterTimeOfDay(test_data, "12:00", "14:00")
    # exit(1)

    sns.kdeplot(data=microsoft["Avg"], cumulative=True, label="Microsoft Avg. RTT")
    sns.kdeplot(data=microsoft["Min"], cumulative=True, label="Microsoft Min. RTT")
    sns.kdeplot(data=microsoft["Max"], cumulative=True, label="Microsoft Max. RTT")

    google = test_data.loc[data["Datacenter Company"] == "GOOGLE"]

    sns.kdeplot(data=google["Avg"], cumulative=True, label="Google Avg. RTT")
    sns.kdeplot(data=google["Min"], cumulative=True, label="Google Min. RTT")
    sns.kdeplot(data=google["Max"], cumulative=True, label="Google Max. RTT")

    amazon = test_data.loc[data["Datacenter Company"] == "AMAZON"]

    sns.kdeplot(data=amazon["Avg"], cumulative=True, label="Amazon Avg. RTT")
    sns.kdeplot(data=amazon["Min"], cumulative=True, label="Amazon Min. RTT")
    sns.kdeplot(data=amazon["Max"], cumulative=True, label="Amazon Max. RTT")

    plt.legend(title="RTT type", loc="upper left")
    plt.xlabel('RTT [ms]')
    plt.ylabel('CDF')
    plt.title("CDF of Ping RTT")
    plt.grid("both")

    exportToPdf(fig, args.output)

    # Trying a boxplot
    fig, ax = plt.subplots()

    test_data = test_data.loc[test_data["Continent"] == "ME"]
    test_data = test_data.loc[test_data["Datacenter Continent"] == "ME"]

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/wifi_me.pdf")

    fig, ax = plt.subplots()

    test_data = data.loc[data["Technology"] == "WIFI"]
    test_data = test_data.loc[data["Avg"] > 0]
    test_data = test_data.loc[test_data["Continent"] == "EU"]
    test_data = test_data.loc[test_data["Datacenter Continent"] == "EU"]

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/wifi_eu.pdf")

    fig, ax = plt.subplots()

    test_data = data.loc[data["Technology"] == "WIFI"]
    test_data = test_data.loc[data["Avg"] > 0]
    test_data = test_data.loc[test_data["Continent"] == "EU"]
    test_data = test_data.loc[test_data["Datacenter Continent"] == "NA"]

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/wifi_eu_na.pdf")

    test_data = data.loc[data["Technology"] == "CELLULAR"]
    test_data = test_data.loc[data["Avg"] > 0]

    fig, ax = plt.subplots()

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/cell.pdf")

    test_data = data.loc[data["Technology"] == "SATELLITE"]
    test_data = test_data.loc[data["Avg"] > 0]

    fig, ax = plt.subplots()

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/sat.pdf")

    test_data = data.loc[data["Technology"] == "LAN"]
    test_data = test_data.loc[data["Avg"] > 0]

    fig, ax = plt.subplots()

    sns.boxplot(data=test_data, x="Continent", y="Avg", hue="Datacenter Company")

    # Styling the plot
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    plt.grid(axis="y")

    exportToPdf(fig, "results/ping/lan.pdf")


# Adapted from: 
# https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
def plotLocationMap(inputs, output):
    """
    Expecting the input file(s) with the continent to ID matching.
    Extracting the latitude and longitude and plotting this data on a map.
    """

    locations = extractGeolocation(inputs, "connected.json")
    with open("test.csv", "w") as test_csv:
        test_csv.write("Longitude,Latitude\n")
        for point in locations:
            test_csv.write(f"{point.x},{point.y}\n")
    
    df = pd.read_csv("test.csv", delimiter=",")
    gdf = GeoDataFrame(df, geometry=locations)
    # gdf = GeoDataFrame(df[:10], geometry=locations[:10])
    # gdf2 = GeoDataFrame(df[10:20], geometry=locations[10:20])

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10,6)), marker='o', color="red", markersize=10, label="Test", legend=True)
    # gdf2.plot(ax=world.plot(figsize=(10,6)), marker='o', color="green", markersize=10, label="Test", legend=True)

    plt.show()

# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate CDF plots to compare the ping latency')
    parser.add_argument('-i','--input', action="append", default=[], required=True, help="The path to the JSON file containg the latency information")
    parser.add_argument('-o','--output', type=str, default="results/ping/pingRTT.pdf", help="The file in which the resulting plot is stored")

    args = parser.parse_args()

    plotPingLatencyCDF(args)
    # plotLocationMap(args.input, "map.pdf")