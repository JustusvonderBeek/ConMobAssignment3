import requests
import json
import os
import concurrent.futures
import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

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

def filterJsonList(input_json, field):
    """
    Expecting a json object containing a list.
    Filtering the json list and returning the values of the given field.
    """

    output = set()
    for elem in input_json:
        output.add(elem[field])

    return list(output)

def combineMeasures(input_json, id_list):
    """
    Expecting a json object list and the individual IDs.
    Combining measures based on the time of the day. E.g. measures at 2 p.m are combined.
    Returning a pandas DataFrame.
    """

    columns = ["Measurement", "Probe ID", "Timestamp", "Sent", "Received", "Latency", "Src", "Dst"]
    data = pd.DataFrame(columns=columns)
    # print(data)

    for id in id_list:
        id_measures = [elem for elem in input_json if elem["prb_id"] == id]
        # printJSON(id_measures)

        datapoints = list()
        for test in id_measures:
            latency = []
            for elem in test["result"]:
                if "rtt" in elem.keys():
                    latency.append(elem["rtt"])
            datapoints.append({"Measurement": "Ping", "Probe ID": id, "Timestamp": test["timestamp"], "Sent": test["sent"], "Received": test["rcvd"], "Latency": latency, "Src": test["src_addr"], "Dst": test["dst_addr"]})

        # print(datapoints)
        data = pd.concat([pd.DataFrame(datapoints), data], ignore_index=True)
        # print(data.to_markdown())

    # print(data.to_markdown())

    return data

def pingMeasureToDataFrame(id):
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
    # print(ping)
    involved_probes = filterJsonList(ping, "prb_id")
    # print(involved_probes)
    data = combineMeasures(ping, involved_probes)

    return data

def rawPingMeasureToCsv(id_list, csv):
    """
    Expecting a list of IDs pointing to all PING measurements.
    Downloading measurement information for all IDs.
    Saving the output of the measurements to the given CSV file.
    """

    # NOTE: Because this method and especially fetching data takes VERY long, parallelize the for loop

    df = None
    # Printing a progress bar
    # In order for output to work correctly, use tqdm.write(<string>) - this should make the output readable
    # and keep the progress bar at the bottom of the terminal
    with tqdm(total=len(id_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_df = {executor.submit(pingMeasureToDataFrame, id): id for id in id_list}
            
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

# -------------------------------------------------------------------sss-------
# MAIN METHOD
# --------------------------------------------------------------------------


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', type=str, default="20230208.json")
    parser.add_argument('-o', '--output', type=str, default="connected.json")
    parser.add_argument('-l', '--list', action="store_true", help="Printing all available user measurements.")
    parser.add_argument('-f', '--filter', action="store_true", help="Filtering node types into the given output file. Overwriting existing files!")
    parser.add_argument('-m', '--matching', action="store_true", help="Matching measurement points from existing *.json files. Overwriting existing files!")

    args = parser.parse_args()

    if args.list:
        ping_ids, trace_ids = listMeasurements()

        rawPingMeasureToCsv(ping_ids, "measurements/ping/ping.csv")