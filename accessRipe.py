import requests
import json
import posixpath
import haversine as hs
import numpy as np

from argparse import ArgumentParser
from urllib.parse import urljoin
from collections import defaultdict
from tqdm import tqdm

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
        # print(f"Wrote filtered connected nodes to '{args.output}'")

    filtered = filterLocalNodes(args.output, "tags", ["system-ipv6-works", "system-ipv4-works"])
    nodeCount = len(filtered["objects"])
    print(f"Nodes matching the filter: {nodeCount}")
    
    with open(args.output, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{args.output}'")

def filterCellular(args):
    """
    Filtering for cellular nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["lte", "mobile", "5g", "4g", "3g", "t-mobile"])
    
    out_filename = "cellular.json"
    with open(out_filename, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{out_filename}'")

def filterWiFi(args):
    """
    Filtering for WiFi nodes and writing these to the given output file.
    """

    filtered = filterLocalNodes(args.input, "tags", ["wi-fi", "wireless", "system-wifi", "fixed-wireless"])

    out_filename = "wifi.json"
    with open(out_filename, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{out_filename}'")

def filterLAN(args):
    """
    Filtering for LAN nodes and writing these to the given output file.
    """

    # Note: 'home' 'dsl' is not necessarily LAN
    filtered = filterLocalNodes(args.input, "tags", ["ftth","fibre","cable","vdsl2","vdsl","adsl","pppoe","google-fiber","fttb-2","fttp-2","fttb"])

    out_filename = "lan.json"
    with open(out_filename, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        print(f"Wrote filtered connected nodes to '{out_filename}'")

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

def findWeirdNodes():
    with open("cellular.json", "r") as input_nodes_cellular:
        cellNodes = json.load(input_nodes_cellular)
    with open("lan.json", "r") as input_nodes_lan:
        lanNodes = json.load(input_nodes_lan)
    with open("wifi.json", "r") as input_nodes_wifi:
        wifiNodes = json.load(input_nodes_wifi)

    nodes0 = cellNodes
    nodes1 = lanNodes
    nodes2 = wifiNodes

    filtered_dict = defaultdict(list)

    # find identical nodes in all sets
    for node0 in nodes0["objects"]:
        id0 = node0["id"]
        
        for node1 in nodes1["objects"]:
            id1 = node1["id"]

            for node2 in nodes2["objects"]:
                id2 = node2["id"]

                if id0 == id1 or id0 == id2:
                    if node0 not in filtered_dict["objects"]:
                        filtered_dict["objects"].append(node0)
                elif id1 == id2:
                    if node1 not in filtered_dict["objects"]:
                        filtered_dict["objects"].append(node1)
        
    print(f"Found {len(filtered_dict['objects'])} nodes with conflictings tags")
    
    with open("weird.json", "w") as output_file:
        json.dump(filtered_dict, output_file, indent=4)


def findTripleMatches():
    with open("cellular.json", "r") as input_nodes_cellular:
        cellNodes = json.load(input_nodes_cellular)
    with open("lan.json", "r") as input_nodes_lan:
        lanNodes = json.load(input_nodes_lan)
    with open("wifi.json", "r") as input_nodes_wifi:
        wifiNodes = json.load(input_nodes_wifi)

    # we can try different ordering
    nodes0 = cellNodes
    nodes1 = lanNodes
    nodes2 = wifiNodes

    meassurement_points = list()

    for node0 in nodes0["objects"]:
        loc0 = makeGeolocation(node0["latitude"], node0["longitude"])
        
        # rank all other nodes by distance to node0
        matches_01 = list()
        matches_02 = list()

        for node1 in nodes1["objects"]:
            loc1 = makeGeolocation(node1["latitude"], node1["longitude"])
            dist01 = getDistance(loc0, loc1)
            matches_01.append({"id0" : node0["id"], "id1" : node1["id"], "dist" : dist01})
        
        for node2 in nodes2["objects"]:
            loc2 = makeGeolocation(node2["latitude"], node2["longitude"])
            dist02 = getDistance(loc0, loc2)
            matches_02.append({"id0" : node0["id"], "id2" : node2["id"], "dist" : dist02})
        
        def sortByDist(e):
            return e['dist']

        matches_01.sort(key=sortByDist)
        matches_02.sort(key=sortByDist)

        assert(matches_01[0]['id0'] == matches_02[0]['id0']) # sanity check

        # take the closests node from each list
        meassurement_points.append({
            'id0' : node0['id'], 
            'id1' : matches_01[0]['id1'], 
            'id2' : matches_02[0]['id2'], 
            'dist01' : matches_01[0]['dist'], 
            'dist02' : matches_02[0]['dist'], 
            'dist' : matches_01[0]['dist'] + matches_02[0]['dist']
            })
        
        # We only want the 'best/closest" 100
        meassurement_points.sort(key=sortByDist)
    
    print(f"meassurement_points[0] = {meassurement_points[0]}")
    print(f"meassurement_points[99] = {meassurement_points[99]}")

def combineOnly2Nodes(args):
    with open("cellular.json", "r") as input_cellular:
        cellularProbes = json.load(input_cellular)
    with open("wifi.json", "r") as input_wifi:
        wifiProbes = json.load(input_wifi)
    with open("lan.json", "r") as input_lan:
        lanProbes = json.load(input_lan)

    cellular_ids,lan_ids = findMatchingNodes(cellularProbes, lanProbes)
    filtered_cellular = filterLocalNodes("connected.json", "id", cellular_ids)

    with open("cellular_test.csv", "w") as output_file:
        output_file.write("Cellular IDs,LAN IDs,Country\n")
        country_codes = getCountryCodes(cellular_ids)
        for i in tqdm(range(len(cellular_ids))):
            output_file.write(str(cellular_ids[i]) + ",")
            output_file.write(str(lan_ids[i]) + ",")
            output_file.write(country_codes[i] + "\n")

    wifi_ids,lan_ids = findMatchingNodes(wifiProbes, lanProbes)
    filtered_cellular = filterLocalNodes("connected.json", "id", wifi_ids)

    with open("wifi_test.csv", "w") as output_file:
        output_file.write("WiFi IDs,LAN IDs,Country\n")
        country_codes = getCountryCodes(wifi_ids)
        for i in tqdm(range(len(wifi_ids))):
            output_file.write(str(wifi_ids[i]) + ",")
            output_file.write(str(lan_ids[i]) + ",")
            output_file.write(country_codes[i] + "\n")
    
    return

def combineNodes(args):

    with open("cellular.json", "r") as input_nodes_base:
        baseNodes = json.load(input_nodes_base)
    with open("wifi.json", "r") as input_nodes_compare:
        possibleNodes = json.load(input_nodes_compare)

    cellular_ids,wifi_ids = findMatchingNodes(baseNodes, possibleNodes)
    filtered = filterLocalNodes("connected.json", "id", cellular_ids)
    # print(filtered)

    with open("lan.json", "r") as input_nodes_lan:
        possibleNodes = json.load(input_nodes_lan)
    _,lan_ids = findMatchingNodes(filtered, possibleNodes)

    # Writing to csv
    print("Resolving Country Code and writing to file...")
    country_codes = getCountryCodes(lan_ids)
    with open(args.output, "w") as output_file:
        output_file.write("Cellular IDs,Wifi IDs,LAN IDs,Country\n")
        for i in tqdm(range(len(cellular_ids))):
            output_file.write(str(cellular_ids[i]) + ",")
            output_file.write(str(wifi_ids[i]) + ",")
            output_file.write(str(lan_ids[i]) + ",")
            output_file.write(country_codes[i] + "\n")

def findMatchingNodes(baseNodes, possibleNodes):
    """
    Expecting two lists of nodes. Matching the two lists to obtain one list of nodes lying within the given threshold range.
    """

    counter = 0
    org_id_list = list()
    matched_id_list = list()
    locations = [ (makeGeolocation(node["latitude"], node["longitude"]),node["id"]) for node in possibleNodes["objects"] ]
    for baseNode in baseNodes["objects"]:
        location = makeGeolocation(baseNode["latitude"], baseNode["longitude"])
        for loc,id in locations:
            if inRange(location, loc):
                # print(f"Found match between: {location} vs. {loc} = {getDistance(location, loc)}")
                locations.remove((loc,id))
                counter += 1
                org_id_list.append(baseNode["id"])
                matched_id_list.append(id)
                break

    print(f"Found {counter} matches")
    # print(id_list)

    return org_id_list,matched_id_list

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

def getCountryCode(id):

    return getCountryCodes([id])[0]

def getCountryCodes(id_list):
    """
    Expecting the probe ID from the RIPE Atlas network.
    Returning the country code.
    """

    country_codes = list()
    with open("connected.json", "r") as input_file:
        probes = json.load(input_file)
        probes = probes["objects"]
        for id in id_list:
            country_code = next((node["country_code"] for node in probes if node["id"] == id), "Unknown")
            country_codes.append(country_code)

    return country_codes

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

    # combineNodes(args)
    combineOnly2Nodes(args)
    
    # findWeirdNodes()
    # findTrippleMatches()


# --------------------------------------------------------------------------
# END OF MAIN
# --------------------------------------------------------------------------