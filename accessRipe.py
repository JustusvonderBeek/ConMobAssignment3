import requests
import json
import posixpath
from argparse import ArgumentParser
from urllib.parse import urljoin
from collections import defaultdict
import haversine as hs

# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

base_url = "https://atlas.ripe.net"
distance_threshold = 100.0 # Distance in kilometers

# --------------------------------------------------------------------------
# REST API ACCESS
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# END OF REST API ACCESS
# --------------------------------------------------------------------------

def filterLocalNodes(input, field, tags):
    """
    Gets the path to the json file containing all nodes. Extracting the nodes matching the list of tags.
    """
    
    filtered_dict = defaultdict(list)
    with open(input, "r") as file:
        nodes = json.load(file)
        for elem in nodes["objects"]:
            if type(elem[field]) == list:
                for tag in tags:
                    if tag in elem[field]:
                        filtered_dict["objects"].append(elem)
                        # print("Found matching element")
                        break
            else:
                for tag in tags:
                    if tag == elem[field]:
                        filtered_dict["objects"].append(elem)
                        # print("Found matching element")
                        break

    return filtered_dict

def filterConnected(args):
    """
    Filtering all nodes for the connected ones and writing these to a special file.
    """

    filtered = filterLocalNodes(args.input, "status_name", ["Connected", "connected"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterCellular(args):
    """
    Filtering for cellular nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["lte", "mobile", "5g", "4g", "3g", "t-mobile"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterWiFi(args):
    """
    Filtering for WiFi nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["wi-fi", "wireless", "system-wifi", "fixed-wireless"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterLAN(args):
    """
    Filtering for LAN nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["home","ftth","fibre","cable","dsl","vdsl2","vdsl","adsl","pppoe","google-fiber","fttb-2","fttp-2","fttb"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def showAvailableTags(args):
    """
    Reading an input file with connected nodes and printing all available user tags which can be filtered for.
    """

    filtered_dict = defaultdict(dict)
    with open(args.input, "r") as input_file:
        nodes = json.load(input_file)
        for node in nodes["objects"]:
            for tag in node["tags"]:
                filtered_dict[tag] = filtered_dict.get(tag, 0) + 1

    for v in sorted(filtered_dict, key=filtered_dict.get, reverse=True):
        print(f"{v}:".ljust(32), f"{filtered_dict[v]}")

# --------------------------------------------------------------------------
# GEOLOCATION CALCULATIONS
# --------------------------------------------------------------------------

def makeGeolocation(lat,lon):
    """
    Input is the latitude and longitude as strings.
    Returns the combined location.
    """

    return (float(lat),float(lon))

def getDistance(loc1, loc2):
    """
    Expecting input of two geo-locations as (lat,lon).
    Returns the distance between the two points in KM.
    """

    hs.haversine(loc1, loc2)

def inRange(loc1,loc2):
    """
    Checking if two geo-locations are within a certain range.
    Returns true if lower than threshold, false otherwise.
    """

    return getDistance(loc1, loc2) < distance_threshold


# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', type=str, default="20230202.json")
    parser.add_argument('-o', '--output', type=str, default="connected.json")
    args = parser.parse_args()

    # filterWifiNodes()
    # filterLocalNodes("20230202.json", "connected.json", ["wi-fi", "wireless", "wifi", "WiFi", "WIFI", "wireless-isp"])

    # showAvailableTags(args)
    # filterConnected(args)

    # filterCellular(args)
    # filterWiFi(args)
    # filterLAN(args)

# --------------------------------------------------------------------------
# END OF MAIN
# --------------------------------------------------------------------------