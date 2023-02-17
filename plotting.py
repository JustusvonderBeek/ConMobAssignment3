import os
import re
import time
import datetime

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

def plotLineplot(data, technology):
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
        wifi = combineAroundTimepoint(filtered, interval="4h")
        sns.lineplot(data=wifi, x="Time", y="Avg", markers=True, label=f"{cont}", color=pal[cont])

    # Styling the graph
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.set_xticklabels(ax.get_xticklabels(), rotation="45", horizontalalignment='right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a - %d.%m.%Y - %H:%M'))
    ax.grid(visible=True, which="major")
    ax.grid(visible=True, which="minor", color="#c9c9c9", linestyle=":")
    ax.set_ylim(ymin=0)
    ax.set_ylabel("RTT [ms]")
    ax.set_xlabel("Timestamp")
    plt.title(f"{technology} Technology")
    plt.legend(title="Continent", loc="upper left")

    exportToPdf(fig, f"results/ping/{technology}_line.pdf", width=8, height=6)

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

def plotPingLatencyLineplot(args):
    """
    Expecting the parsed command line containing input and output file information.
    Parsing the measurement results and plotting the CDF for latencies.
    """

    print('Plotting Ping Latency Lineplot')

    data = pd.read_csv(args.input[0], na_filter=False)
    
    plotLineplot(data, "WIFI")

    plotLineplot(data, "CELLULAR")

    plotLineplot(data, "SATELLITE")

    plotLineplot(data, "LAN")


def plotLatencyDifferences(inputs):
    """
    Plotting the differences between TODO
    Saving to TODO
    """

    technology = "Satellite"
    technology_upper = technology.upper()
    technology_lower = technology.lower()
    data = pd.read_csv(inputs[0], na_filter=False)
    # Load the matching
    matching = pd.read_csv(f"measurement_creation/{technology_lower}_lan_match.csv", index_col=None, na_filter=True)
    recap_matching = matching.copy()
    out_matching = None
    wifi_ids = [x for x in matching[f"{technology}"] if not np.isnan(x) ]
    lan_ids = [int(x) for x in matching["Lan"] if not np.isnan(x) ]

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
        missing_counter = 0
        missing_ids = set()
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
                    missing_counter += 1
                    missing_ids.add(id)
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
        if len(avg_lan) < len(avg_wifi):
            for i in range(len(avg_lan)):
                matching.loc[matching["Lan"] == avg_lan[i][0], "Lan Avg"] = avg_lan[i][1]
                matching.loc[matching["Lan"] == avg_lan[i][0], "Continent"] = avg_lan[i][2]
                matching.loc[matching["Lan"] == avg_lan[i][0], "Datacenter"] = avg_lan[i][3]
                matching_wifi = matching.loc[matching["Lan"] == avg_lan[i][0], f"{technology}"].values[0]
                wifi_elem = next((x for x in avg_wifi if x[0] == matching_wifi), None)
                if len(wifi_elem) is None:
                    print("Did not find match!!!!!")
                    exit(1)
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

    # print(f"Matching? {matching.loc[matching['Lan'] == avg_lan[0][0]]}")

    fig, axes = plt.subplots()
    sns.boxplot(data=out_matching, x="Continent", y="Diff", hue="Datacenter")

    axes.set_xlabel("Continents")
    axes.set_ylabel(f"{technology} RTT - Lan RTT [ms]")
    plt.title(f"RTT Difference between LAN and {technology}")
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
    with open("test.csv", "w") as test_csv:
        test_csv.write("Longitude,Latitude\n")
        for point in locations:
            test_csv.write(f"{point.x},{point.y}\n")
    
    df = pd.read_csv("test.csv", delimiter=",")
    # Remove the temporary file
    os.remove("test.csv")
    gdf = GeoDataFrame(df, geometry=locations)
    # gdf = GeoDataFrame(df[:10], geometry=locations[:10])
    # gdf2 = GeoDataFrame(df[10:20], geometry=locations[10:20])

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = gdf.plot(ax=world.plot(figsize=(10,6)), marker='o', cmap="plasma", markersize=6, label="Test", legend=True)
    # gdf2.plot(ax=world.plot(figsize=(10,6)), marker='o', color="green", markersize=10, label="Test", legend=True)

    # Styling (remove ticks)
    ax.set_axis_off()
    # plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    # plt.show()
    # Saving to file
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

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10,6)), marker='o', color="#e83b14", markersize=6, label="Test", legend=True)

    # Styling (remove ticks)
    plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
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
    parser.add_argument('-i','--input', action="append", default=[], required=True, help="The path to the JSON file containg the latency information")
    parser.add_argument('-o','--output', type=str, default="results/ping/pingRTT.pdf", help="The file in which the resulting plot is stored")

    args = parser.parse_args()

    # plotPingLatencyBoxplot(args)
    # plotPingInterLatencyBoxplot(args)
    plotPingLatencyLineplot(args)
    # plotLatencyDifferences(args.input)
    # plotProbeLocationMap(args.input, "results/probemap.pdf")
    # plotDatacenterLocationMap(args.input)