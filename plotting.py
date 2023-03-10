import os
import re
import time
import datetime
import statistics
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json, scipy, ast
import geopandas as gpd
import matplotlib.dates as mdates

from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from shapely.geometry import Point
from geopandas import GeoDataFrame, tools
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def exportToPdf(fig, filename, width=8, height=6):
    """
    Exports the current plot to file. Both in the 10:6 and 8:6 format (for thesis and slides.
    """

    # Saving to 8:6 format (no name change)
    fig.set_figwidth(width)
    fig.set_figheight(height)

    # Check if folder exists and create the path if failed
    dirname, fname = os.path.split(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig.savefig(filename, bbox_inches='tight', format='pdf')
    print(f"Wrote output to '{filename}'")

def filterJsonList(input_json, field):
    """
    Expecting a json object containing a list.
    Filtering the json list and returning the values of the given field.
    """

    output = set()
    for elem in input_json:
        if field in elem.keys():
            output.add(elem[field])

    return list(output)

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

    # Extract the technology of the probe
    with open("probes/lan.json", "r") as lan:
        lan_probes = json.load(lan)
    with open("probes/cellular.json", "r") as cellular:
        cell_probes = json.load(cellular)
    with open("probes/satellite.json", "r") as sat:
        sat_probes = json.load(sat)
    with open("probes/wifi.json", "r") as wifi:
        wifi_probes = json.load(wifi)

    lan_ids = filterJsonList(lan_probes["objects"], "id")
    cell_ids = filterJsonList(cell_probes["objects"], "id")
    sat_ids = filterJsonList(sat_probes["objects"], "id")
    wifi_ids = filterJsonList(wifi_probes["objects"], "id")

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
            if id in cell_ids:
                technology = "CELLULAR"
            elif id in sat_ids:
                technology = "SATELLITE"
            elif id in wifi_ids:
                technology = "WIFI"
            else:
                technology = "LAN"
            location = next(((node["longitude"],node["latitude"]) for node in probes if node["id"] == id), "Unknown")
            locations.append((Point(location), technology))

    # print(locations)
    return locations

def convertCityToLocation(input):
    """
    Expecting the datacenter file with City names.
    Fetching the geolocation and returning a list of latitudes and longitudes.
    """

    df = pd.read_csv(input)
    cities = list(df["Location"].unique())
    locations = gpd.tools.geocode(cities)

    # print(f"{locations}")

    return locations


def parseTime(timestamp):
    """
    Expecting a string timestamp like '10:12'.
    Returning datetime
    """
    hour = int(re.findall("([\d]+):", timestamp)[0])
    min = int(re.findall(":([\d]+)", timestamp)[0])
    return datetime.time(hour=hour, minute=min)

def timeInRange(test_time, start, end):
    # test_time = pd.to_datetime(test_time, unit="s")
    if start <= end:
        if start.hour == test_time.hour:
            return start.minute <= test_time.minute
        elif end.hour == test_time.hour:
            return test_time.minute <= end.minute
        return start.hour < test_time.hour < end.hour
    else:
        if start.hour == test_time.hour:
            return start.minute <= test_time.minute
        elif end.hour == test_time.hour:
            return test_time.hour <= end.minute
        return start.hour <= test_time.hour or test_time.hour <= end.hour

def filterTimeOfDay(dataframe, start, end):
    """
    Expecting the PING measurement data and the time interval which should be filtered.
    Returning the filtered DataFrame where PINGs happened only within the interval.
    """

    start = parseTime(start)
    end = parseTime(end)
    # print(f"{start.hour}")

    df = dataframe.copy()
    df["Time"] = pd.to_datetime(dataframe["Timestamp"], unit="s")
    # print(f"{dataframe['Time']}")
    df = df.loc[df["Time"].apply(timeInRange, args=(start, end))]
    # print(f"{dataframe.to_markdown()}")

    return df

def convertStringDayToWeekday(day):
    """
    Converts a string day to a python time_struct weekday number.
    Necessary because the output of "%a" in gmtime depends on the system language!
    """
    if day == "Mon":
        return 0
    elif day == "Tue":
        return 1
    elif day == "Wed":
        return 2
    elif day == "Thu":
        return 3
    elif day == "Fri":
        return 4
    elif day == "Sat":
        return 5
    elif day == "Sun":
        return 6
    else:
        print(f"Unknown day {day}!")
    return -1

def filterDays(dataframe, days):
    """
    Expecting the PING measurement data and a list of weekdays in string format, e.g. 'Sun'.
    weekdays = ["Mon", "Tue", "Thu", "Wed", "Fri", "Sat", "Sun"]
    Returning the filtered DataFrame.
    """

    df = dataframe
    df["Weekday"] = df["Timestamp"].apply(lambda x: time.gmtime(x).tm_wday)
    # print(f"{dataframe['Weekday']}")
    days = list(map(convertStringDayToWeekday, days))
    # print(f"{days}")
    df = df.loc[df["Weekday"].apply(lambda x: x in days)]
    # print(f"{dataframe.to_markdown()}")

    return df

def filterInvalid(dataframe):
    """
    This method can be used to filter invalid PINGs from the DataFrame.
    """

    # Currently we only filter pings that did not reached the destination
    return dataframe.loc[dataframe["Avg"] > 0].copy()

def filterAccessTechnology(dataframe, technology):
    """
    Expecting the PING measurement data and the access technology.
    Returning the filtered DataFrame where PINGs used the given technology.
    """

    return dataframe.loc[dataframe["Technology"] == technology].copy()

def filterIntraContinent(dataframe, continent):
    """
    Expecting the PING measurement data and the continent as input.
    Returning the filtered DataFrame where connections only go from CONTINENT -> CONTINENT.
    """

    df = dataframe.loc[dataframe["Continent"] == continent]
    return df.loc[df["Continent"] == df["Datacenter Continent"]]

def filterInterContinent(dataframe, source, dest):
    """
    Expecting the PING measurement data and the source and destination continent as input.
    Returning the filtered DataFrame where connections go from SOURCE -> DEST.
    """

    df = dataframe.loc[dataframe["Continent"] == source].copy()
    return df.loc[df["Datacenter Continent"] == dest]

def filterDaytimeWeekday(dataframe):
    dataframe = filterDays(dataframe, ["Mon", "Tue", "Wed", "Thu", "Fri"])
    dataframe = filterTimeOfDay(dataframe, "08:00", "20:00")
    return dataframe

def filterNighttimeWeekday(dataframe):
    dataframe = filterDays(dataframe, ["Mon", "Tue", "Wed", "Thu", "Fri"])
    dataframe = filterTimeOfDay(dataframe, "20:00", "08:00")
    return dataframe

def filterDaytimeWeekend(dataframe):
    dataframe = filterDays(dataframe, ["Sat", "Sun"])
    dataframe = filterTimeOfDay(dataframe, "08:00", "20:00")
    return dataframe

def filterNighttimeWeekend(dataframe):
    dataframe = filterDays(dataframe, ["Sat", "Sun"])
    dataframe = filterTimeOfDay(dataframe, "20:00", "08:00")
    return dataframe

def combineAroundTimepoint(dataframe, start="2023-02-09 19:00:00", interval="2h"):
    """
    Expecting the pre-filtered DataFrame. Combining PING measures at any given day in the interval.
    Returning the combined DataFrame.
    """

    dataframe["Time"] = pd.to_datetime(dataframe["Timestamp"], unit="s")
    groupby = dataframe.groupby(pd.Grouper(key="Time", freq=f"{interval}", origin=f"{start}"))
    dataframe = groupby.mean(numeric_only=True)

    # print(f"{dataframe.to_markdown()}")

    return dataframe

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------

def plotQuadAccessTechnology(raw_data, technology, prefix=""):
    """
    Expecting an access technology like "WIFI" and the filtered data on this technology data["Technology"] == "<Tech>"
    Plotting and saving the technology to disk.
    """

    weekday_day = filterDaytimeWeekday(raw_data)

    # PLOTTING
    fig, axes = plt.subplots(2,2)
    sns.boxplot(data=weekday_day, x="Continent", y="Avg", hue="Datacenter Company", ax=axes[0,0])
    
    # Styling the plot
    axes[0,0].set_axisbelow(True)
    axes[0,0].set_ylim(ymin=0, ymax=800)
    axes[0,0].grid(axis="y")
    axes[0,0].legend(title="Datacenter", loc="upper right")
    axes[0,0].set_title(f"{technology} Ping Latency (Mon-Fri, 08:00-20:00)")

    weekday_night = filterNighttimeWeekday(raw_data)

    sns.boxplot(data=weekday_night, x="Continent", y="Avg", hue="Datacenter Company", ax=axes[0,1])

    axes[0,1].grid(axis="y")
    axes[0,1].set_axisbelow(True)
    axes[0,1].set_ylim(ymin=0, ymax=800)
    axes[0,1].legend(title="Datacenter", loc="upper right")
    axes[0,1].set_title(f"{technology} Ping Latency (Mon-Fri, 20:00-08:00)")

    weekend_day = filterDaytimeWeekend(raw_data)

    sns.boxplot(data=weekend_day, x="Continent", y="Avg", hue="Datacenter Company", ax=axes[1,0])

    axes[1,0].set_axisbelow(True)
    axes[1,0].set_ylim(ymin=0, ymax=800)
    axes[1,0].grid(axis="y")
    axes[1,0].legend(title="Datacenter", loc="upper right")
    axes[1,0].set_title(f"{technology} Ping Latency (Sat-Sun, 08:00-20:00)")

    weekend_night = filterNighttimeWeekend(raw_data)

    sns.boxplot(data=weekend_night, x="Continent", y="Avg", hue="Datacenter Company", ax=axes[1,1])

    axes[1,1].set_axisbelow(True)
    axes[1,1].set_ylim(ymin=0, ymax=800)
    axes[1,1].grid(axis="y")
    axes[1,1].legend(title="Datacenter", loc="upper right")
    axes[1,1].set_title(f"{technology} Ping Latency (Sat-Sun, 20:00-08:00)")

    exportToPdf(fig, f"results/ping/{prefix}{technology}.pdf", width=16, height=12)

def plotLineplot(data, technology, ymax=200):
    continents = ["EU", "NA", "SA", "AS", "AF", "OC", "ME"]
    # Perform some pre-filtering
    valid_pings = filterInvalid(data)
    valid_pings = valid_pings.loc[valid_pings["Continent"] == valid_pings["Datacenter Continent"]]
    # Comparing Wifi Across the Globe with LAN
    tech = filterAccessTechnology(valid_pings, technology)

    fig, ax = plt.subplots()
    # From our measurement creation we started roughly around 18:40 at 09.02
    # That means we gather 19:00 , 21:00 , 23:00, 01:00 , ... etc.
    # The interval goes 18:00 - 20:00

    pal = {"EU":"tab:blue", "OC":"tab:orange", "NA": "tab:green", "SA": "tab:red", "AF": "tab:purple", "AS": "tab:brown", "ME": "tab:pink"}
    for cont in continents:
        filtered = tech.loc[tech["Continent"] == cont].copy()
        if len(filtered) == 0:
            continue
        # print(f"{cont} - {technology}: {len(filtered)/3}")
        wifi = combineAroundTimepoint(filtered, interval="4h")
        sns.lineplot(data=wifi, x="Time", y="Avg", markers=True, label=f"{cont}", color=pal[cont])

    # Styling the graph
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.set_xticklabels(ax.get_xticklabels(), rotation="45", horizontalalignment='right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a - %d.%m.%Y - %H:%M'))
    ax.grid(visible=True, which="major")
    ax.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":")
    ax.set_ylim(ymin=0, ymax=ymax)
    ax.set_ylabel("RTT [ms]")
    ax.set_xlabel("Timestamp [UTC]")
    if technology == "LAN":
        technology = "Wired"
    plt.title(f"{string.capwords(technology)} Technology")
    plt.legend(title="Continent", loc="upper left")

    exportToPdf(fig, f"results/ping/{technology.lower()}_line.pdf", width=12, height=6)

def plotPingLatencyBoxplot(args):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency Boxplot')

    # data = extractPingLatencies(args)
    data = pd.read_csv(args.input[0], na_filter=False)

    # Perform some pre-filtering
    valid_pings = filterInvalid(data)
    # Comparing Wifi Across the Globe with LAN
    valid_pings = valid_pings.loc[valid_pings["Continent"] == valid_pings["Datacenter Continent"]]
    wifi = filterAccessTechnology(valid_pings, "WIFI")
    
    plotQuadAccessTechnology(wifi, "WIFI")

    cell = filterAccessTechnology(valid_pings, "CELLULAR")
    plotQuadAccessTechnology(cell, "CELLULAR")
    
    sat = filterAccessTechnology(valid_pings, "SATELLITE")
    plotQuadAccessTechnology(sat, "SATELLITE")
    
    lan = filterAccessTechnology(valid_pings, "LAN")
    plotQuadAccessTechnology(lan, "LAN")
    
def plotPingInterLatencyBoxplot(args):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency Inter Continent Boxplot')

    # data = extractPingLatencies(args)
    data = pd.read_csv(args.input[0], na_filter=False)

    # Perform some pre-filtering
    valid_pings = filterInvalid(data)
    # Comparing Wifi Across the Globe with LAN
    valid_pings = valid_pings.loc[valid_pings["Continent"] != valid_pings["Datacenter Continent"]]
    
    wifi = filterAccessTechnology(valid_pings, "WIFI")
    plotQuadAccessTechnology(wifi, "WIFI", prefix="inter_")

    cell = filterAccessTechnology(valid_pings, "CELLULAR")
    plotQuadAccessTechnology(cell, "CELLULAR", prefix="inter_")
    
    sat = filterAccessTechnology(valid_pings, "SATELLITE")
    plotQuadAccessTechnology(sat, "SATELLITE", prefix="inter_")
    
    lan = filterAccessTechnology(valid_pings, "LAN")
    plotQuadAccessTechnology(lan, "LAN", prefix="inter_")

def plotPingLatencyLineplot(input):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency Lineplot')

    data = pd.read_csv(input[0], na_filter=False)
    
    plotLineplot(data, "WIFI")

    plotLineplot(data, "CELLULAR", ymax=350)

    plotLineplot(data, "SATELLITE")

    plotLineplot(data, "LAN")


def plotLatencyDifferences(inputs):
    """
    Plotting the differences between access technologies ordered by continent
    Saving to ping _lan_match.csv files.
    """

    technology = "Wifi"
    technology_upper = technology.upper()
    technology_lower = technology.lower()
    data = pd.read_csv(inputs[0], na_filter=False)
    # Load the matching
    matching = pd.read_csv(f"measurement_creation/{technology_lower}_lan_match.csv", index_col=None, na_filter=True)
    recap_matching = matching.copy()
    out_matching = None
    wifi_ids = [x for x in matching[f"{technology}"] if not np.isnan(x) ]
    lan_ids = [int(x) for x in matching["Lan"] if not np.isnan(x) ]
    wifi_ids = wifi_ids[:len(lan_ids)]

    # print(f"Length:\nWifi:{len(wifi_ids)}\nLan:{len(lan_ids)}")
    # print(f"Wifi: {wifi_ids}\nLan: {lan_ids}")

    # Preparing the data by removing invalid and filter for inter-continent, wifi
    # print(f"Raw data:\n{data.loc[:,'Latency':'Datacenter Continent'].to_markdown()}")
    valid_pings = filterInvalid(data)
    # print(f"Invalid filtered data:\n{valid_pings.loc[:,['Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")
    valid_pings = valid_pings.loc[valid_pings["Continent"] == valid_pings["Datacenter Continent"]]
    # print(f"Filtered intra continent:\n{valid_pings.loc[:,['Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")

    datacenters = ["GOOGLE", "AMAZON", "MICROSOFT"]
    # datacenter = datacenters[2]

    for datacenter in datacenters:
        matching = recap_matching
        local_v_pings = valid_pings.loc[valid_pings["Datacenter Company"] == datacenter]
        # print(f"Filtered company:\n{valid_pings.loc[:,['Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")
        if len(local_v_pings) == 0:
            continue

        wifi = filterAccessTechnology(local_v_pings, f"{technology_upper}")
        # print(f"Filtered wifi:\n{wifi.loc[:,['Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")

        wifi = wifi.loc[wifi["Probe ID"].apply(lambda id: id in wifi_ids)]
        # print(f"Filtered on IDs:\n{wifi.loc[:,['Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")

        lan = filterAccessTechnology(valid_pings, "LAN")
        lan = lan.loc[lan["Probe ID"].apply(lambda id: id in lan_ids)]
        # print(f"Filtered on IDs:\n{lan.loc[:,['Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")

        # print(f"Length: {len(wifi.index)} vs. {len(lan.index)}")

        # For each continent
        avg_wifi = list()
        contained_ids = set()
        continents = ["EU", "NA", "SA", "AS", "AF", "OC", "ME"]
        for continent in continents:
            print(f"Continent: {continent}")
            wifi_cont = filterIntraContinent(wifi, continent)
            # print("Filtered INTRA Continent:\n" + wifi_cont.loc[:,['Probe ID','Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown())
            for id in wifi_ids:
                wifi_filtered = wifi_cont.loc[wifi_cont["Probe ID"] == id]
                # print(f"Filtered for {id}:\n{wifi_filtered.loc[:,['Probe ID', 'Technology', 'Latency', 'Avg', 'Continent', 'Datacenter Company', 'Datacenter Continent']].to_markdown()}")
                # This is actually A LOT! Can happen because we are filtering for continents right now and so many of the probe IDs wont match
                if len(wifi_filtered) == 0:
                    # print("Missing wifi filtered")
                    continue
                # print(f"{wifi_filtered.to_markdown()}")
                avg = wifi_filtered.loc[:,"Avg"].mean()
                avg_wifi.append((id, avg, continent, datacenter))
                contained_ids.add(id)

        # print(f"Missing Probes: {len(list(missing_ids))}")
        # print(f"List of missing IDs: {list(missing_ids)}")
        # print(f"Avg latencies: {avg_wifi}")

        # Should be working now: Test and analyze the input (filtered) to the loop.... might be that the data contains many surprises

        # Filtering the Lan IDs that match to the contained wifi IDs
        lan_ids = list()
        for id in contained_ids:
            lan_id = matching.loc[matching[f"{technology}"] == id]
            # print(lan_id["Lan"].values[0])
            if np.isnan(lan_id["Lan"].values[0]):
                continue
            lan_ids.append(int(lan_id["Lan"].values[0]))

        # print(lan_ids)
        avg_lan = list()
        for continent in continents:
            lan_cont = filterIntraContinent(lan, continent)

            for id in lan_ids:
                lan_filtered = lan_cont.loc[lan_cont["Probe ID"] == id]

                if len(lan_filtered) == 0:
                    # print("Missing LAN probe in measurement")
                    continue
                    
                avg = lan_filtered.loc[:,"Avg"].mean()
                avg_lan.append((id, avg, continent, datacenter))

        # print(f"Avg lan latencies: {avg_lan}")
        # Matching into the dataframe
        # if len(avg_lan) < len(avg_wifi):
        for i in range(len(avg_lan)):
            matching.loc[matching["Lan"] == avg_lan[i][0], "Lan Avg"] = avg_lan[i][1]
            matching.loc[matching["Lan"] == avg_lan[i][0], "Continent"] = avg_lan[i][2]
            matching.loc[matching["Lan"] == avg_lan[i][0], "Datacenter"] = avg_lan[i][3]
            matching_wifi_id = matching.loc[matching["Lan"] == avg_lan[i][0], f"{technology}"].values[0]
            wifi_elem = next((x for x in avg_wifi if x[0] == matching_wifi_id), None)
            if wifi_elem is None or len(wifi_elem) is None:
                # print("Did not find match!!!!!")
                matching.loc[matching["Lan"] == avg_lan[i][0], f"{technology} Avg"] = np.nan
                # exit(1)
                continue
            matching.loc[matching["Lan"] == avg_lan[i][0], f"{technology} Avg"] = wifi_elem[1]

        # print(matching.to_markdown())

        print("Computing the difference")

        matching["Diff"] = matching[f"{technology} Avg"] - matching["Lan Avg"]
        matching = matching[matching["Diff"].notnull()]

        # print(matching.to_markdown())

        if out_matching is None:
            out_matching = matching
        else:
            out_matching = pd.concat([out_matching, matching], ignore_index=True)
            out_matching.to_csv(f"measurements/ping/matched_pings_{technology_lower}.csv")
        
        
        # matching.to_csv("measurements/ping/matched_pings_wifi.csv")

    v = out_matching.Continent.value_counts()
    out_matching = out_matching[out_matching.Continent.isin(v.index[v.gt(8)])]

    # print(f"Matching? {matching.loc[matching['Lan'] == avg_lan[0][0]]}")

    fig, axes = plt.subplots()
    sns.boxplot(data=out_matching, x="Continent", y="Diff", hue="Datacenter")

    axes.set_xlabel("Continents")
    axes.set_ylabel(f"{technology} RTT - Wired RTT [ms]")
    plt.title(f"RTT Difference between {technology} and Wired")
    axes.yaxis.get_ticklocs(minor=True)
    axes.minorticks_on()
    axes.set_axisbelow(True)
    axes.grid(visible=True, which="major", axis="y")
    axes.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":", axis="y")
    plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)

    # plt.show()
    exportToPdf(fig, f"results/ping/{technology_lower}_lan_diff.pdf", width=8, height=6)

    fig, axes = plt.subplots()

    # Sanity check plot
    sns.countplot(data=out_matching, x="Continent", hue="Datacenter")

    axes.set_xlabel("Continents")
    axes.set_ylabel("# Measures")
    plt.title(f"# Measures per Continent and Datacenter ({technology_upper})")
    axes.yaxis.get_ticklocs(minor=True)
    axes.minorticks_on()
    axes.set_axisbelow(True)
    axes.grid(visible=True, which="major", axis="y")
    axes.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":", axis="y")

    exportToPdf(fig, f"results/ping/{technology_lower}_lan_count.pdf", width=8, height=6)

def plotTracerouteBarplot(input, output):
    """
    Expecting the traceroute data file location and the output location.
    Plotting the path as Piechart and storing the result at the output location.
    """

    data = pd.read_csv(input, index_col=None, na_filter=False)
    data = data.loc[data["Prb Continent"] == data["Datacenter Continent"]]

    datacenters = ["GOOGLE", "AMAZON", "MICROSOFT"]
    # continents = ["EU", "NA", "SA", "AS", "AF", "OC", "ME"]
    continents = ["EU", "NA"]

    for continent in tqdm(continents, desc="Plotting Continents"):
        for datacenter in datacenters:
            iter_data = data.loc[data["Prb Continent"] == continent]
            iter_data = iter_data.loc[iter_data["Datacenter Company"] == datacenter]
            # print(iter_data.to_markdown())

            asn_list = list()
            comp_list = list()
            for index,row in iter_data.iterrows():
                asn_list.extend(ast.literal_eval(row["ASN"]))
                comp_list.extend(ast.literal_eval(row["ASN Company"]))

            dataframe = pd.DataFrame({"ASN":asn_list, "Organization":comp_list, "Occurrence":np.ones(len(asn_list))})
            dataframe = dataframe.value_counts(subset=["Organization"], sort=True).rename_axis('Organization').to_frame('Occurrence').reset_index(level=0, inplace=False)

            # print(f"{dataframe.to_markdown()}")
            dataframe.to_csv(f"measurements/traceroute/asn_occ_{continent.lower()}_{datacenter.lower()}.csv", index=None)
            # exit(1)

            dataframe.loc[:,"Organization"] = dataframe["Organization"].apply(lambda x: re.findall("[A-Z0-9\-]+", x)[0])
            df_draw = dataframe.copy()
            df_draw.loc[df_draw['Occurrence'] < 10, 'Organization'] = 'Other (< 10 Occurrences)'
            df_draw.loc[df_draw["Organization"] == "NA","Organization"] = "Local IPs / Not Responding"
            dataframe = df_draw.groupby('Organization')['Occurrence'].sum().reset_index()
            
            dataframe = dataframe.sort_values(by="Occurrence", ascending=False)
            # dataframe.to_csv("measurements/traceroute/piechart.csv", index=None)

            # print(f"{dataframe.to_markdown()}")

            fig, ax = plt.subplots()
            sns.barplot(data=dataframe, x="Occurrence", y="Organization", palette="cool")

            # Styling the plot
            def log2_base(y, pos):
                return r"${{ {:d} }}$".format(int(y))
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(log2_base))
            ax.minorticks_on()
            ax.set_axisbelow(True)
            ax.set_xlim(xmin=1)
            ax.set_ylabel("Organization / ASN")
            ax.grid(visible=True, which="major", axis="x")
            ax.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":", axis="x")
            plt.tick_params(axis='y', which='minor', left=False, bottom=False, top=False, labelbottom=False)
            plt.title(f"{continent} Intra-Continental ASN Occurrence on Path to {datacenter} Datacenter")

            # plt.show()

            exportToPdf(fig, f"{output}asn_{continent}_{datacenter}.pdf", width=10, height=6)

def plotTraceCDF(input, output):
    """
    Expecting the path to the traceroute CSV.
    Reading the file and extracting the average latency.
    Plotting a CDF for this latency and saving to the location given under output.
    """

    org_data = pd.read_csv(input, index_col=None, na_filter=False)

    technologies = ["LAN", "SATELLITE", "CELLULAR"]
    technologies = technologies[:2]
    continents = ["EU", "NA", "AS", "OC", "AF", "SA", "ME"]
    continents = continents[0:2] # Only plot until NA for now

    continent_matching = defaultdict(list)
    continent_matching["EU"]=["EU","NA", "AF", "AS", "ME"]
    continent_matching["NA"]=["NA","EU", "SA", "AS", "OC"]
    continent_matching["AS"]=["NA","EU", "SA", "OC"]
    continent_matching["OC"]=["NA", "SA", "ME", "AS"]
    continent_matching["SA"]=["NA", "AF", "AS", "OC"]
    continent_matching["AF"]=["EU", "SA", "ME"]
    continent_matching["ME"]=["EU", "AF", "OC"]

    df = None
    for continent in tqdm(continents, desc="Plotting Continents"):
        source = continent
        for dest in continent_matching[continent][:3]:
            # Select the first match for now
            # dest = continent_matching[continent][0]
            for technology in technologies:
                # technology = "LAN"

                # data = data.loc[data["Prb Continent"] != data["Datacenter Continent"]]
                data = org_data.loc[org_data["Prb Continent"] == source]
                data = data.loc[data["Datacenter Continent"] == dest]
                # We only pinged a single datacenter, so filtering for this is not necessary
                # data = data.loc[data["Datacenter Company"] == "AMAZON"]
                # print(f"{data.to_markdown()}")

                data = data.loc[data["Technology"] == technology]
                # print(len(data))

                latencies = list()
                for row in data.iterrows():
                    row = row[1]
                    lat = ast.literal_eval(row["Latency Avg"])
                    # for elem in lat:
                        # elem *= 1e3
                        # latencies.append(elem)        
                    latencies.extend(lat)

                # print(min(latencies))
                source_col = [source] * len(latencies)
                dest_col = [dest] * len(latencies)
                if technology == "LAN":
                    technology = "Wired"
                technology = string.capwords(technology)
                tech_col = [technology] * len(latencies)
            
                df = pd.concat([df, pd.DataFrame({"Source": source_col, "Destination":dest_col, "Technology":tech_col, "Latency":latencies})], ignore_index=True)

    # print(f"{df.to_markdown()}")
    # print(latencies)
    fig, ax = plt.subplots()
    # sns.kdeplot(data=latencies, cumulative=True, label=f"Test")
    # plt.hist(latencies, density=True, bins=len(latencies), cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
    # sns.histplot(latencies, stat="count", cumulative=True)
    # plt_data = df.loc[df["Technology"] == "SATELLITE"]
    plt_data = df.copy()
    # plt_data = plt_data.loc[plt_data["Source"] == "NA"]
    # plt_data = plt_data.loc[plt_data["Destination"] == "EU"]
    hue = plt_data[['Technology', 'Source', 'Destination']].apply(lambda row: f"{row.Technology}, {row.Source}, {row.Destination}", axis=1)
    # The title of the legend
    hue.name = "Technology, Source, Destination"
    sns.ecdfplot(data=plt_data, x="Latency", hue=hue)
    upper_limit = statistics.stdev(plt_data["Latency"])
    upper_limit = 150
    # print(upper_limit)
    # plt_data = df.copy()
    # plt_data = plt_data.loc[plt_data["Source"] == "EU"]
    # plt_data = plt_data.loc[plt_data["Destination"] == "EU"]
    # sns.ecdfplot(data=plt_data, x="Latency", log_scale=False, hue="Technology")
    # upper_limit = max(upper_limit, statistics.stdev(plt_data["Latency"]))

    # plt_data = df.loc[df["Technology"] == "LAN"]
    # sns.ecdfplot(data=plt_data, x="Latency", log_scale=False, label="Lan")
    # n_upper_limit = statistics.stdev(plt_data["Latency"])
    # upper_limit = max(upper_limit, n_upper_limit)

    # Styling the plot
    # ax.set_xscale("log")
    ax.set_xlim(xmin=-5, xmax=2.6 * upper_limit)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(visible=True, which="major", axis="both")
    ax.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":", axis="both")
    # plt.tick_params(axis='y', which='minor', left=False, bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("CDF")
    ax.set_xlabel("RTT [ms]")
    plt.title(f"Inter-Continental vs. Intra-Continental Latency CDF")
    # plt.show()

    exportToPdf(fig, f"{output}cdf_{source.lower()}_{dest.lower()}_{technology.lower()}.pdf")


def plotMobileTraceCDF(input, output):
    """
    Plotting CDFs for mobile clients intra-continental and inter-continental.
    Saving output to the folder given in output.
    """

    org_data = pd.read_csv(input, index_col=None, na_filter=False)

    dest = "EU"
    org_data = org_data.loc[org_data["Datacenter Continent"] == dest]
    
    technologies = ["WIFI", "CELLULAR", "SATELLITE"]
    # technologies = technologies[:2]
    sources = ["DE", "FR", "US"]

    df = None
    for technology in technologies:
        for source in sources:
            data = org_data.loc[org_data["Technology"] == technology]
            # print(f"Before {len(data)}")
            data = data.loc[data["Prb Country"] == source]
            # print(f"After {len(data)}")
            # print(data.to_markdown())

            latencies = list()
            for row in data.iterrows():
                row = row[1]
                lat = ast.literal_eval(row["Latency Avg"])
                # for elem in lat:
                    # elem *= 1e3
                    # latencies.append(elem)        
                latencies.extend(lat)

            # print(min(latencies))
            source_col = [source] * len(latencies)
            dest_col = [dest] * len(latencies)
            tech = string.capwords(technology)
            tech_col = [tech] * len(latencies)
        
            df = pd.concat([df, pd.DataFrame({"Source": source_col, "Destination":dest_col, "Technology":tech_col, "Latency":latencies})], ignore_index=True)
    

    # print(df.to_markdown())

    fig, ax = plt.subplots()
    plt_data = df.copy()
    
    # print(max(df["Latency"]))

    hue = plt_data[['Technology', 'Source', 'Destination']].apply(lambda row: f"{row.Technology}, {row.Source}, DE", axis=1)
    # The title of the legend
    hue.name = "Technology, Source, Destination"
    sns.ecdfplot(data=plt_data, x="Latency", hue=hue)
    
    # Cutting the upper limit (tail)
    upper_limit = statistics.stdev(plt_data["Latency"])
    upper_limit = 100

    # Adding a vertical line
    plt.axvline(30, color="#262626", linestyle="--")
    plt.axvspan(-5,30, color="#d3d3d3B0")

    # Styling the plot
    # ax.set_xscale("log")
    ax.set_xlim(xmin=-5, xmax=2.6 * upper_limit)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(visible=True, which="major", axis="both")
    ax.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":", axis="both")
    # plt.tick_params(axis='y', which='minor', left=False, bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("CDF")
    ax.set_xlabel("RTT [ms]")
    plt.title(f"Wireless Access Technology CDF for Different Pathlengths")
    # plt.show()

    exportToPdf(fig, f"{output}cdf_wireless_to_eu.pdf")

# --------------------------------------------------------------------------
# PLOTTING THE MAPS
# --------------------------------------------------------------------------


# Adapted from: 
# https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
def plotProbeLocationMap(inputs, output):
    """
    Expecting the input file(s) with the continent to ID matching.
    Extracting the latitude and longitude and plotting this data on a map.
    """

    locations = extractGeolocation(inputs, "probes/connected.json")
    # with open("test.csv", "w") as test_csv:
    #     test_csv.write("Longitude,Latitude\n")
    #     for point in locations:
    #         test_csv.write(f"{point.x},{point.y}\n")
    
    # df = pd.read_csv("test.csv", delimiter=",")
    # Remove the temporary file
    # os.remove("test.csv")

    # Convert the list of (long,lat,tech) into dataframe
    df = pd.DataFrame(locations, columns=["Location","Technology"])
    # print(df.to_markdown())

    wifi = df.loc[df["Technology"] == "WIFI"]
    wifi_probes = GeoDataFrame(wifi, geometry=wifi["Location"])
    lan = df.loc[df["Technology"] == "LAN"]
    lan_probes = GeoDataFrame(lan, geometry=lan["Location"])
    sat = df.loc[df["Technology"] == "SATELLITE"]
    sat_probes = GeoDataFrame(sat, geometry=sat["Location"])
    cell = df.loc[df["Technology"] == "CELLULAR"]
    cell_probes = GeoDataFrame(cell, geometry=cell["Location"])

    fix, ax = plt.subplots()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Create the world without any points
    base = world.plot(color="#009cf7", edgecolor="#d6d6d635", figsize=(12,6), zorder=1)
    
    # Now adding the different probe types on top
    ax = lan_probes.plot(ax=base, zorder=2, marker='o', color="tab:green", markersize=5, label="Wired", legend=True)
    ax = wifi_probes.plot(ax=base, zorder=3, marker='o', color="tab:orange", markersize=5, label="Wifi", legend=True)
    ax = cell_probes.plot(ax=base, zorder=4, marker='o', color="tab:red", markersize=5, label="Cellular", legend=True)
    ax = sat_probes.plot(ax=base, zorder=5, marker='o', color="tab:purple", markersize=5, label="Satellite", legend=True)

    # Styling (remove ticks)
    ax.set_axis_off()
    # Saving to file
    plt.legend(title="Probe Type", loc="center left")
    plt.title("Probe Spread")
    plt.savefig(output, bbox_inches='tight', format='pdf')
    print(f"Wrote output to '{output}'")

def plotDatacenterLocationMap(inputs, output="results/datacentermap.pdf"):
    """
    Expecting the datacenters as input. Fetching the geolocation per City Name.
    Plotting and saving a world map with the given locations of datacenters marked.
    """

    # NOTE: This whole method can timeout. Simply retry and it SHOULD work!

    locations = convertCityToLocation(inputs[0])
    df = pd.DataFrame(locations)
    df = df["geometry"]
    print(df.to_markdown())
    
    gdf = GeoDataFrame(df, geometry=df)

    fig, ax = plt.subplots()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    base = world.plot(color="#009cf7", edgecolor="#d6d6d635", figsize=(12,6), zorder=1)
    ax = gdf.plot(ax=base, zorder=2, marker='o', color="#e83b14", markersize=5, label="Test", legend=True)

    # Styling (remove ticks)
    ax.set_axis_off()
    # plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    # plt.show()
    # Saving to file
    plt.title("Datacenter Locations")
    plt.savefig(output, bbox_inches='tight', format='pdf')
    print(f"Wrote output to '{output}'")

# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate CDF plots to compare the ping latency')
    # parser.add_argument('-i','--input', action="append", default=[], required=True, help="The path to the JSON file containg the latency information")
    # parser.add_argument('-o','--output', type=str, default="results/ping/pingRTT.pdf", help="The file in which the resulting plot is stored")
    parser.add_argument('-p', '--ping', action="store_true", help="Plotting all PING plots")
    parser.add_argument('-m', '--map', action="store_true", help="Plotting all MAP plots")
    parser.add_argument('-t', '--traceroute', action="store_true", help="Plotting all TRACEROUTE plots")
    parser.add_argument('-a', '--all', action="store_true", help="Performing all actions. Fetching all measurements, pings and traceroutes.")

    args = parser.parse_args()

    # Plotting the pings
    if args.ping or args.all:
        # Not used in the analysis
        # plotPingLatencyBoxplot(args)
        # plotPingInterLatencyBoxplot(args)
        
        plotPingLatencyLineplot(["measurements/ping/ping.csv"])
        plotLatencyDifferences(["measurements/ping/ping.csv"])
    
    # Plotting the maps
    if args.map or args.all:
        plotProbeLocationMap(["measurement_creation/continent_v4.csv"], "results/probemap.pdf")
        plotDatacenterLocationMap(["measurement_creation/datacenters.csv"])

    # # Plotting the traceroute
    if args.traceroute or args.all:
        plotTracerouteBarplot("measurements/traceroute/trace.csv", "results/trace/")
        # Use with great care! This function needs to be adapted manually to produce other results!
        plotTraceCDF("measurements/traceroute/trace.csv", "results/trace/")
        plotMobileTraceCDF("measurements/traceroute/trace.csv", "results/trace/")