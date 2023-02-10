import requests
import json

from argparse import ArgumentParser

# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

base_url = "https://atlas.ripe.net/api/v2/"
api_key = "f9a735fa-c429-409d-b028-1050a3ee840b"

curl_measurement_template = """curl -H "Authorization: Key KEY" -X VERB URL"""
curl_template = """curl --dump-header - -H "Content-Type: application/json" -H "Accept: application/json" -X VERB -d '{}' URL"""


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

    with open("ping_measures.json", "w") as output:
        json.dump(ping, output, indent=4)

    access_url = base_url + "measurements/traceroute/?description=CMB Group 9 Part 2"
    traceroute = requests.get(access_url, headers={"Authorization" : "Key " + api_key})
    traceroute = traceroute.json()

    with open("traceroute_measure.json", "w") as output:
        json.dump(traceroute, output, indent=4)

    ping_ids = [ measure["id"] for measure in ping["results"] ]
    traceroute_ids = [ measure["id"] for measure in traceroute["results"] ]

    print(f"PINGS: {json.dumps(ping_ids, indent=4)}")
    print(f"TRACEROUTES: {json.dumps(traceroute_ids, indent=4)}")

    return ping_ids, traceroute_ids


# --------------------------------------------------------------------------
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
        listMeasurements()