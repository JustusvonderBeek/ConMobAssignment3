import requests
import json
import os
import concurrent.futures
import ast
import socket
import ipaddress
import statistics
import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from cymruwhois import Client

# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

base_url = "https://atlas.ripe.net/api/v2/"
api_key = "c5895aa0-4476-4430-91c5-fb821f0377c5"

curl_measurement_template = """curl -H "Authorization: Key KEY" -X VERB URL"""
curl_template = """curl --dump-header - -H "Content-Type: application/json" -H "Accept: application/json" -X VERB -d '{}' URL"""

# The default folder prefix
ping_prefix = "measurements/ping"
trace_prefix = "measurements/traceroute"

# TImeout in seconds
default_timeout = 20

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

    # Load the list of IP to ASNs and add the IPSubnetMask
    # ip_to_asn = pd.read_csv("measurements/traceroute/ip2asn-v4.tsv", sep="\t",)
    # ip_to_asn.columns = ["Start", "End", "ASN", "Country", "Company"]


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
        # "ip2asn": ip_to_asn,
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

    ping_measurement_path = os.path.join(ping_prefix, "ping_measures.json")
    if os.path.exists(ping_measurement_path) and not args.overwrite:
        with open(ping_measurement_path, "r") as ping_input:
            ping = json.load(ping_input)
    else:
        access_url = base_url + "measurements/ping/?description=CMB Group 9 Part 2"
        ping = requests.get(access_url, headers={"Authorization" : "Key " + api_key})
        ping = ping.json()
        # print(answer)

        with open(ping_measurement_path, "w") as output:
            print(f"Wrote ping measurement ID list to '{ping_measurement_path}'")
            json.dump(ping, output, indent=4)

    trace_measurement_path = os.path.join(trace_prefix, "traceroute_measure.json")
    if os.path.exists(trace_measurement_path) and not args.overwrite:
        with open(trace_measurement_path, "r") as trace_input:
            traceroute = json.load(trace_input)
    else:
        access_url = base_url + "measurements/traceroute/?description=CMB Group 9 Part 2"
        traceroute = requests.get(access_url, headers={"Authorization" : "Key " + api_key})
        traceroute = traceroute.json()

        with open(trace_measurement_path, "w") as output:
            print(f"Wrote traceroute measurement ID list to '{trace_measurement_path}'")
            json.dump(traceroute, output, indent=4)

    ping_ids = filterJsonList(ping["results"], "id")
    traceroute_ids = filterJsonList(traceroute["results"], "id")

    if not args.list and not args.overwrite:
        return ping_ids, traceroute_ids

    print(f"PINGS: {json.dumps(ping_ids, indent=4)}")
    print(f"TRACEROUTES: {json.dumps(traceroute_ids, indent=4)}")

    return ping_ids, traceroute_ids

# --------------------------------------------------------------------------
# PING MEASUREMENT
# --------------------------------------------------------------------------

def addMeasureInformation(id, dst, file_dict):
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
    # print(f"ID: {id}")
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
        technology, location, country, continent, dc, dc_continent = addMeasureInformation(id, id_measures[0]["dst_addr"], file_dict)
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
            return None

    # Opening relevant files and creating lists for adding detailed information
    # This should increase the performance because it has to be done only once
    file_dict = openAuxiliaryFiles()

    df = None
    # Printing a progress bar
    # In order for output to work correctly, use tqdm.write(<string>) - this should make the output readable
    # and keep the progress bar at the bottom of the terminal
    # id_list = id_list[15:16]
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

def extractTraceMeasureInformation(traceroute, measurement_id, index, file_dict):
    """
    Expecting the trace measurement JSON and the involved probes.
    Returning a valid DataFrame containg the measurement information.
    """

    involved_probes = filterJsonList(traceroute, "prb_id")
    columns = ["Probe ID", "Technology", "Prb Country", "Prb Continent", "Datacenter IP", "Datacenter Company", "Datacenter Continent", "Hopcount", "Latency Avg", "IP", "ASN", "ASN Company"]
    data = pd.DataFrame(columns=columns)
    # traceroute = traceroute[:10]
    # ip2asn = file_dict["ip2asn"]

    # print(f"Length of JSON: {len(traceroute)}")

    df = None
    # for trace in tqdm(traceroute):
    # for trace in traceroute:
    def inner_loop(trace):
        c = Client()
        # For each IP address that we find in the measurement
        probe_id = trace["prb_id"]
        ips = list()
        asn = list()
        asn_comp = list()
        lat = list()
        last_lat = 0
        hopcount = 0
        for hop in trace["result"]:
            hopcount += 1
            # Check if we got a response from the hop
            # if "x" in hop["result"][0].keys():
            #     print(f"Hop {hop['hop']} did not contain information")
            #     continue
            # Resolve IP to ASN number and ASN Owner
            ip_set = set()
            asn_set = set()
            owner_set = set()
            curr_avg = list()
            for h in hop["result"]:
                if "from" in h.keys():
                    ip_set.add(h["from"])
                if "rtt" in h.keys():
                    curr_avg.append(h["rtt"])

            avg = last_lat
            if len(curr_avg) > 0:
                avg = statistics.mean(curr_avg)
                last_lat = avg

            # for elem in ip_set:
            #     print(f"Elem: {elem}")
            #     ip_int = int(ipaddress.IPv4Address(elem))
            #     bigger_rows = ip2asn.loc[ip2asn["Start"].apply(lambda x: int(ipaddress.IPv4Address(x)) >= ip_int)]
            #     row = bigger_rows.loc[ip2asn["End"].apply(lambda x: int(ipaddress.IPv4Address(x)) <= ip_int)]
            #     if len(row) > 1:
            #         print(f"Oh oh! You matched more than one ASN to the IP: {elem}! What a pitty")
            #         exit(1)
            #     print(f"Matching: {row.to_markdown()}")
            #     asn_set.add(int(row["ASN"]))
            #     owner_set.add(row["Company"])
            
            # The old but slow solution
            for elem in ip_set:
                resp = c.lookup(elem)
                asn_set.add(resp.asn)
                owner_set.add(resp.owner)
            
            ips.extend(list(ip_set))
            asn.extend(list(asn_set))
            asn_comp.extend(list(owner_set))
            lat.append(avg)

        # print(f"IPs: {ips}")
        # print(f"ASNs: {asn}")
        # print(f"ASN-Companies: {asn_comp}")

        technology, location, country, continent, dc_comp, dc_continent = addMeasureInformation(probe_id, trace["dst_addr"], file_dict)

        information = {"Probe ID": probe_id, "Technology": technology, "Prb Country": country, "Prb Continent": continent, "Datacenter IP": trace["dst_addr"], "Datacenter Company": dc_comp, "Datacenter Continent": dc_continent, "Hopcount": hopcount, "Latency Avg": [lat], "IP": [ips], "ASN": [asn], "ASN Company": [asn_comp]}

        # data = pd.concat([pd.DataFrame(information), data], ignore_index=True)

        return pd.DataFrame(information)

    with tqdm(total=len(traceroute), position=index, leave=True, desc=f"TRACE {measurement_id}") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_df = {executor.submit(inner_loop, trace): trace for trace in traceroute}
            
            for future in concurrent.futures.as_completed(future_to_df):
                id = future_to_df[future]
                try:
                    dataframe = future.result()
                except Exception as exc:
                    print(f"ID {id} generated exception: {exc}")
                else:
                    df = pd.concat([dataframe, df], ignore_index=True)
                    pbar.update(1)

    # print(f"{data.to_markdown()}")

    return df

def traceMeasureToDataFrame(measurement_id, index, file_dict):
    """
    Expecting a traceroute measurement ID.
    Returning a Data Frame containing all measurements for this ID.
    """

    access_url = base_url + "measurements/"
    id_url = access_url + str(measurement_id) + "/results/"
    tqdm.write(f"Fetching TRACEROUTE measurements for {measurement_id}...")
    # This can take quite some time (timeout, if none is given the request will not timeout)
    # See: https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
    # traceroute = requests.get(id_url, timeout=default_timeout)
    with open(f"measurements/traceroute/first_traces/trace_{measurement_id}.json") as json_file:
        traceroute = json.load(json_file)
    
    # traceroute = traceroute.json()
    # with open(f"measurements/traceroute/first_traces/trace_{measurement_id}.json", "w") as output:
    #     json.dump(traceroute, output, indent=2)
        # print("Wrote intermediate JSON to 'measurements/traceroute/tmp.json'")
    # printJSON(trace)
    # involved_probes = filterJsonList(trace, "prb_id")
    # print(involved_probes)
    data = extractTraceMeasureInformation(traceroute, measurement_id, index, file_dict)
    # data.to_csv(f"measurements/traceroute/trace_{measurement_id}.csv", index=False)

    return data


def rawTraceMeasurementsToCsv(id_list, csv):
    """
    Expecting the list of all traceroute measurements.
    Fetching the measurement information and storing the measurement in the given CSV file.
    """

    if not args.overwrite:
        if os.path.exists(csv):
            return None

    file_dict = openAuxiliaryFiles()

    df = None
    # id_list = id_list[0:1]
    with tqdm(total=len(id_list), position=0, leave=True, desc="Overall") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=46) as executor:
            future_to_df = {executor.submit(traceMeasureToDataFrame, id, index + 1, file_dict): id for index,id in enumerate(id_list)}
            
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
    print(f"Saved traceroute measurement output to '{csv}'")

    return df

# -------------------------------------------------------------------sss-------
# MAIN METHOD
# --------------------------------------------------------------------------


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-l', '--list', action="store_true", help="Printing all available user measurements and exit.")
    parser.add_argument('-p', '--ping', action="store_true", help="Fetching all ping measurements.")
    parser.add_argument('-t', '--traceroute', action="store_true", help="Fetching all traceroute measurements.")
    parser.add_argument('-a', '--all', action="store_true", help="Performing all actions. Fetching all measurements, pings and traceroutes.")
    parser.add_argument('-w', '--overwrite', action="store_true", help="If existing results should be overwritten!")

    global args
    args = parser.parse_args()

    ping_ids, trace_ids = listMeasurements()
    
    if args.list and not args.all:
        # Only creating measurement listing
        exit(0)
        
    if args.ping or args.all:
        rawPingMeasureToCsv(ping_ids, "measurements/ping/ping.csv")

    if args.traceroute or args.all:
        rawTraceMeasurementsToCsv(trace_ids, "measurements/traceroute/trace.csv")