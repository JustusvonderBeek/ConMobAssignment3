import requests
import json
import posixpath
from argparse import ArgumentParser
from urllib.parse import urljoin
from collections import defaultdict
import haversine as hs
import numpy as np

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
        json.dump(response.json(), file, indent=4)

def filterWifiNodes():
    """
    Accessing the RIPE Atlas network and filtering for WiFi nodes.
    """

    tags = "?tags=DSL"
    rq_url = posixpath.join("/api/v2/probes/", tags)

    response = requests.get(urljoin(base_url, rq_url))
    # print(response.json())

    with open("probes.json", "w") as file:
        json.dump(response.json(), file, indent=4)

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
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterCellular(args):
    """
    Filtering for cellular nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["lte", "mobile", "5g", "4g", "3g", "t-mobile"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterWiFi(args):
    """
    Filtering for WiFi nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["wi-fi", "wireless", "system-wifi", "fixed-wireless"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterLAN(args):
    """
    Filtering for LAN nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["home","ftth","fibre","cable","dsl","vdsl2","vdsl","adsl","pppoe","google-fiber","fttb-2","fttp-2","fttb"])

    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
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

def combineNodes(args):

    with open("cellular.json", "r") as input_nodes_base:
        baseNodes = json.load(input_nodes_base)
    with open("wifi.json", "r") as input_nodes_compare:
        possibleNodes = json.load(input_nodes_compare)

    id_list = findMatchingNodes(baseNodes, possibleNodes)
    filtered = filterLocalNodes("connected.json", "id", id_list)
    # print(filtered)

    with open("lan.json", "r") as input_nodes_lan:
        possibleNodes = json.load(input_nodes_lan)
    id_list = findMatchingNodes(filtered, possibleNodes)

    with open(args.output, "w") as output_file:
        for id in id_list:
            output_file.write(str(id) + "\n")

def findMatchingNodes(baseNodes, possibleNodes):
    """
    Expecting two lists of nodes. Matching the two lists to obtain one list of nodes lying within the given threshold range.
    """

    counter = 0
    id_list = list()
    locations = [ (makeGeolocation(node["latitude"], node["longitude"]),node["id"]) for node in possibleNodes["objects"] ]
    for baseNode in baseNodes["objects"]:
        location = makeGeolocation(baseNode["latitude"], baseNode["longitude"])
        for loc,id in locations:
            if inRange(location, loc):
                # print(f"Found match between: {location} vs. {loc} = {getDistance(location, loc)}")
                locations.remove((loc,id))
                counter += 1
                id_list.append(id)
                break

    print(f"Found {counter} matches")
    # print(id_list)

    return id_list

    # Below matches less (47)

    # counter = 0
    # id_list = list()
    # locations = [ makeGeolocation(node["latitude"], node["longitude"]) for node in possibleNodes["objects"] ]
    # for baseNode in baseNodes["objects"]:
    #     location = makeGeolocation(baseNode["latitude"], baseNode["longitude"])
    #     closest = None
    #     for loc in locations:
    #         if closest is None and inRange(location, loc):
    #             closest = loc
    #         if closest is not None and getDistance(location, loc) < getDistance(location, closest):
    #             closest = loc

    #     if closest is None:
    #         # print("No match in range found")
    #         continue

    #     print(f"Found match between: {location} vs. {closest} = {getDistance(location, closest)}")
    #     locations.remove(closest)
    #     counter += 1

    # print(f"Found {counter} matches")

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

    return hs.haversine(loc1, loc2)

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

    combineNodes(args)

# --------------------------------------------------------------------------
# END OF MAIN
# --------------------------------------------------------------------------