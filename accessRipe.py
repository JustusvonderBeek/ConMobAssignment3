import requests
import json
import posixpath
from argparse import ArgumentParser
from urllib.parse import urljoin
from collections import defaultdict

base_url = "https://atlas.ripe.net"

def filterLanNodes():
    """
    Accessing the RIPE Atlas network and filtering for LAN nodes.
    """

    tags = "?tags=dsl,cable,lan,fibre,vdsl,vdsl2"

    response = requests.get(urljoin(base_url, "/api/v2/probes/"))
    print(response.json())

    with open("lan_probes.json", "w") as file:
        json.dump(response.json(), file)

def filterWifiNodes():
    """
    Accessing the RIPE Atlas network and filtering for WiFi nodes.
    """

    tags = "?tags=DSL"
    rq_url = posixpath.join("/api/v2/probes/", tags)

    response = requests.get(urljoin(base_url, rq_url))
    # print(response.json())

    with open("probes.json", "w") as file:
        json.dump(response.json(), file)

def filterLocalNodes(input, output, field, tags):
    """
    Gets the path to the json file containing all nodes. Extracting the nodes matching the list of tags.
    """
    
    filtered_dict = defaultdict(list)
    with open(input, "r") as file:
        nodes = json.load(file)
        for elem in nodes["objects"]:
            for tag in tags:
                if elem[field] == tag:
                    filtered_dict["objects"].append(elem)
                    # for tag in tags:
                    #     if tag in elem["tags"]:
                    #         filtered.append(elem["id"])

    # with open(output, "w") as output_file:
        # json.dump(filtered_dict, output_file, indent=2)

    return filtered_dict

def filterConnected(args):
    """
    Filtering all nodes for the connected ones and writing these to a special file.
    """

    filtered = filterLocalNodes(args.input, "connected.json", "status_name", ["Connected", "connected"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file)
        print(f"Wrote filtered connected nodes to '{args.output}'")


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', type=str, default="20230202.json")
    parser.add_argument('-o', '--output', type=str, default="connected.json")
    args = parser.parse_args()

    # filterWifiNodes()
    # filterLocalNodes("20230202.json", "connected.json", ["wi-fi", "wireless", "wifi", "WiFi", "WIFI", "wireless-isp"])

    filterConnected(args)