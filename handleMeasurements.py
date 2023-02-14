import requests
import json
import os
import concurrent.futures
import ast
import socket
import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from cymruwhois import Client

# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

base_url = "https://atlas.ripe.net/api/v2/"
api_key = "f9a735fa-c429-409d-b028-1050a3ee840b"

curl_measurement_template = """curl -H "Authorization: Key KEY" -X VERB URL"""
curl_template = """curl --dump-header - -H "Content-Type: application/json" -H "Accept: application/json" -X VERB -d '{}' URL"""

# The default folder prefix
ping_prefix = "measurements/ping"
trace_prefix = "measurements/traceroute"

# TImeout in seconds
default_timeout = 10

# --------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# --------------------------------------------------------------------------

def printJSON(input_json):
    """
    Expecting valid JSON input in object format.
    Printing the json in a human readable form.
    """

    print(f"{json.dumps(input_json, indent=2)}")

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

def openAuxiliaryFiles():
    """
    Opening all probe files and extracting the raw JSON.
    Returning a dict pointing to the raw JSON and the probe ID lists which specify which probe belongs to which type.
    """

    with open("probes/lan.json", "r") as lan:
        lan_probes = json.load(lan)
    with open("probes/cellular.json", "r") as cellular:
        cell_probes = json.load(cellular)
    with open("probes/satellite.json", "r") as sat:
        sat_probes = json.load(sat)
    with open("probes/wifi.json", "r") as wifi:
        wifi_probes = json.load(wifi)

    continent = pd.read_csv("measurement_creation/continent_v4.csv", index_col=None, na_filter=False)
    # print(continent.to_markdown())
    datacenters = pd.read_csv("measurement_creation/datacenters.csv", index_col=None, na_filter=False)

    # print(f"{datacenters['IP']}")
    # print(f"{datacenters.loc[datacenters['IP'] == '20.113.132.119', 'Continent'].values[0]}")
    # exit(1)

    lan_ids = filterJsonList(lan_probes["objects"], "id")
    cell_ids = filterJsonList(cell_probes["objects"], "id")
    sat_ids = filterJsonList(sat_probes["objects"], "id")
    wifi_ids = filterJsonList(wifi_probes["objects"], "id")

    file_dict = {
        "lan_prbs": lan_probes,
        "cell_prbs": cell_probes,
        "sat_prbs": sat_probes,
        "wifi_prbs": wifi_probes,
        "lan_ids": lan_ids,
        "cell_ids": cell_ids,
        "sat_ids": sat_ids,
        "wifi_ids": wifi_ids,
        "EU": ast.literal_eval(list(continent.loc[continent["Continent"] == "EU"]["IDs"])[0]),
        "NA": ast.literal_eval(list(continent.loc[continent["Continent"] == "NA"]["IDs"])[0]),
        "SA": ast.literal_eval(list(continent.loc[continent["Continent"] == "SA"]["IDs"])[0]),
        "AS": ast.literal_eval(list(continent.loc[continent["Continent"] == "AS"]["IDs"])[0]),
        "AF": ast.literal_eval(list(continent.loc[continent["Continent"] == "AF"]["IDs"])[0]),
        "OC": ast.literal_eval(list(continent.loc[continent["Continent"] == "OC"]["IDs"])[0]),
        "ME": ast.literal_eval(list(continent.loc[continent["Continent"] == "ME"]["IDs"])[0]),
        "datacenters": datacenters,
    }

    return file_dict

# --------------------------------------------------------------------------
# LISTING ALL MEASUREMENTS
# --------------------------------------------------------------------------

def listMeasurements():
    """
    This method will access the measurements available under the given API key.
    It prints all measurement IDs.
    Returns two lists, the list of all ping IDs and the list of all traceroute IDs.
    """

    access_url = base_url + "measurements/ping/?description=CMB Group 9 Part 2"
    ping = requests.get(access_url, headers={"Authorization" : "Key " + api_key})
    ping = ping.json()
    # print(answer)

    with open(os.path.join(ping_prefix, "ping_measures.json"), "w") as output:
        json.dump(ping, output, indent=4)

    access_url = base_url + "measurements/traceroute/?description=CMB Group 9 Part 2"
    traceroute = requests.get(access_url, headers={"Authorization" : "Key " + api_key})
    traceroute = traceroute.json()

    with open(os.path.join(trace_prefix, "traceroute_measure.json"), "w") as output:
        json.dump(traceroute, output, indent=4)

    ping_ids = [ measure["id"] for measure in ping["results"] ]
    traceroute_ids = [ measure["id"] for measure in traceroute["results"] ]

    # print(f"PINGS: {json.dumps(ping_ids, indent=4)}")
    # print(f"TRACEROUTES: {json.dumps(traceroute_ids, indent=4)}")

    return ping_ids, traceroute_ids

# --------------------------------------------------------------------------
# PING MEASUREMENT
# --------------------------------------------------------------------------

def addPingInformation(id, dst, file_dict):
    """
    Expecting a single measurement probe and the destination IP.
    Returning Technology, Probe Location, Probe Country, Probe Continent, DC Company, DC Continent
    """

    def filterProbe(probe_list):
        for elem in probe_list:
            if elem["id"] == id:
                return elem

    # Find the probe and access type
    if id in file_dict["cell_ids"]:
        technology = "CELLULAR"
        probe = filterProbe(file_dict["cell_prbs"]["objects"])
    elif id in file_dict["sat_ids"]:
        technology = "SATELLITE"
        probe = filterProbe(file_dict["sat_prbs"]["objects"])
    elif id in file_dict["wifi_ids"]:
        technology = "WIFI"
        probe = filterProbe(file_dict["wifi_prbs"]["objects"])
    else:
        technology = "LAN"
        probe = filterProbe(file_dict["lan_prbs"]["objects"])

    # Extracting probe location information
    loc = (probe["latitude"], probe["longitude"])
    country = probe["country_code"]

    # Find the probe continent
    continent = "Unknown"
    if id in file_dict["EU"]:
        continent = "EU"
    elif id in file_dict["NA"]:
        continent = "NA"
    elif id in file_dict["SA"]:
        continent = "SA"
    elif id in file_dict["AS"]:
        continent = "AS"
    elif id in file_dict["AF"]:
        continent = "AF"
    elif id in file_dict["OC"]:
        continent = "OC"
    elif id in file_dict["ME"]:
        continent = "ME"

    # Find datacenter information
    datacenters = file_dict["datacenters"]
    row = datacenters.loc[datacenters["IP"] == str(dst)]
    dc_url = str(row["Datacenter"])
    if ".ec2." in dc_url:
        dc_comp = "AMAZON"
    elif ".gce." in dc_url:
        dc_comp = "GOOGLE"
    elif ".azure." in dc_url:
        dc_comp = "MICROSOFT"
    
    dc_cont = str(row["Continent"].values[0])
    
    return technology, loc, country, continent, dc_comp, dc_cont
    

def extractPingMeasureInformation(input_json, id_list, measure_id, file_dict):
    """
    Expecting a json object list and the individual probe IDs.
    Combining measures based on the time of the day. E.g. measures at 2 p.m are combined.
    Returning a pandas DataFrame.
    """

    columns = ["Measurement", "Measurement ID", "Probe ID", "Technology", "Timestamp", "Sent", "Received", "Latency", "Min", "Max", "Avg", "Src", "Dst", "Location", "Country", "Continent", "Datacenter Company", "Datacenter Continent"]
    data = pd.DataFrame(columns=columns)
    # print(data)

    for id in id_list:
        id_measures = [elem for elem in input_json if elem["prb_id"] == id]
        technology, location, country, continent, dc, dc_continent = addPingInformation(id, id_measures[0]["dst_addr"], file_dict)
        # printJSON(id_measures)

        datapoints = list()
        for test in id_measures:
            latency = filterJsonList(test["result"], "rtt")
            datapoints.append({"Measurement": "Ping", "Measurement ID": measure_id, "Probe ID": id, "Technology": technology, "Timestamp": test["timestamp"], "Sent": test["sent"], "Received": test["rcvd"], "Latency": latency, "Min": test["min"], "Max": test["max"], "Avg": test["avg"], "Src": test["src_addr"], "Dst": test["dst_addr"], "Location": location, "Country": country, "Continent": continent, "Datacenter Company": dc, "Datacenter Continent": dc_continent})

        # print(datapoints)
        data = pd.concat([pd.DataFrame(datapoints), data], ignore_index=True)
        # print(data.to_markdown())

    # print(data.to_markdown())

    return data

def pingMeasureToDataFrame(id, file_dict):
    """
    Expecting a ping measurement ID.
    Returning a Data Frame containing all measurements for this ID.
    """

    access_url = base_url + "measurements/"
    id_url = access_url + str(id) + "/results/"
    tqdm.write(f"Fetching PING measurements for {id}...")
    # This can take quite some time (timeout, if none is given the request will not timeout)
    # See: https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
    ping = requests.get(id_url, timeout=default_timeout)
    ping = ping.json()
    # if id == 49857256:
    #     tqdm.write(f"{len(ping)}")
    #     with open("measurements/ping/large_measure.json", "w") as json_dump:
    #         json.dump(ping, json_dump, indent=2)
    # print(ping)
    involved_probes = filterJsonList(ping, "prb_id")
    # print(involved_probes)
    data = extractPingMeasureInformation(ping, involved_probes, id, file_dict)

    return data

def rawPingMeasureToCsv(id_list, csv):
    """
    Expecting a list of IDs pointing to all PING measurements.
    Downloading measurement information for all IDs.
    Saving the output of the measurements to the given CSV file.
    """

    # NOTE: Because this method and especially fetching data takes VERY long, parallelize the for loop

    if not args.overwrite:
        if os.path.exists(csv):
            return pd.read_csv(csv, index_col=None)

    # Opening relevant files and creating lists for adding detailed information
    # This should increase the performance because it has to be done only once
    file_dict = openAuxiliaryFiles()

    df = None
    # Printing a progress bar
    # In order for output to work correctly, use tqdm.write(<string>) - this should make the output readable
    # and keep the progress bar at the bottom of the terminal
    with tqdm(total=len(id_list), desc="Fetching PINGs") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_df = {executor.submit(pingMeasureToDataFrame, id, file_dict): id for id in id_list}
            
            for future in concurrent.futures.as_completed(future_to_df):
                id = future_to_df[future]
                try:
                    dataframe = future.result()
                except Exception as exc:
                    print(f"ID {id} generated exception: {exc}")
                else:
                    df = pd.concat([dataframe, df], ignore_index=True)
                    pbar.update(1)

    df.to_csv(csv, index=False)
    print(f"Saved ping measurement output to '{csv}'")

    return df

# --------------------------------------------------------------------------
# TRACEROUTE MEASUREMENT
# --------------------------------------------------------------------------

def extractTraceMeasureInformation(trace, prbs, file_dict):
    """
    Expecting the trace measurement JSON and the involved probes.
    Returning a valid DataFrame containg the measurement information.
    """

    domain = "google.de"
    ip = socket.gethostbyname(domain)
    c = Client()
    r = c.lookup(ip)
    print(f"Owner: {r.owner}")

    return None

def traceMeasureToDataFrame(id, file_dict):
    """
    Expecting a traceroute measurement ID.
    Returning a Data Frame containing all measurements for this ID.
    """

    access_url = base_url + "measurements/"
    id_url = access_url + str(id) + "/results/"
    tqdm.write(f"Fetching TRACEROUTE measurements for {id}...")
    # This can take quite some time (timeout, if none is given the request will not timeout)
    # See: https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
    trace = requests.get(id_url, timeout=default_timeout)
    trace = trace.json()
    printJSON(trace)
    involved_probes = filterJsonList(trace, "prb_id")
    print(involved_probes)
    data = extractTraceMeasureInformation(trace, involved_probes, file_dict)

    return data


def rawTraceMeasurementsToCsv(id_list, csv):
    """
    Expecting the list of all traceroute measurements.
    Fetching the measurement information and storing the measurement in the given CSV file.
    """

    df = None
    for id in id_list[:1]:
        traceMeasureToDataFrame(id, None)

    df.to_csv(csv, index=False)
    print(f"Saved traceroute measurement output to '{csv}'")

    return df

# -------------------------------------------------------------------sss-------
# MAIN METHOD
# --------------------------------------------------------------------------


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', type=str, default="20230208.json")
    parser.add_argument('-o', '--output', type=str, default="connected.json")
    parser.add_argument('-l', '--list', action="store_true", help="Printing all available user measurements.")
    parser.add_argument('-w', '--overwrite', action="store_true", help="If existing results should be overwritten!")
    # parser.add_argument('-m', '--matching', action="store_true", help="Matching measurement points from existing *.json files. Overwriting existing files!")

    global args
    args = parser.parse_args()

    if args.list:
        ping_ids, trace_ids = listMeasurements()
        # Extract the PING CSV Table; already stores the result in the given file
        ping_data = rawPingMeasureToCsv(ping_ids, "measurements/ping/ping.csv")

        # TODO: Fetch traceroute measurements
        # trace_data = rawTraceMeasurementsToCsv(trace_ids, "measurements/trace/trace.csv")