import requests
import json
import posixpath
import csv
import ast
import haversine as hs
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from urllib.parse import urljoin
from collections import defaultdict
from tqdm import tqdm


# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

base_url = "https://atlas.ripe.net"
distance_threshold = 100.0 # Distance in kilometers

cellular_tags = ["lte", "5g", "4g", "3g"]
wifi_tags = ["wifi", "wi-fi", "wireless", "system-wifi", "fixed-wireless"]

default_measurement_script = "measurement_creation/measurements.sh"

# --------------------------------------------------------------------------
# PROBE FILTERING
# --------------------------------------------------------------------------

def filterLocalNodes(input, field, tags):
    """
    Gets the path to the json file containing all nodes. 
    Extracting the nodes matching the list of tags.

    :param input: A string to the file containing all probes
    :param field: The object field that should be matched
    :param tags: The tag(s) which should be matched
    :return: Returning a dictionary of all nodes matching the filter.
    """

    filtered_dict = defaultdict(list)
    with open(input, "r") as file:
        nodes = json.load(file)
        for elem in nodes["objects"]:
            if type(elem[field]) == list:
                if any(set(tags).intersection(elem[field])):
                    filtered_dict["objects"].append(elem)
            else:
                if elem[field] in tags:
                    filtered_dict["objects"].append(elem)

    return filtered_dict


def filterLocalNodesNegated(input, field, tags):
    """
    Gets the path to the json file containing all nodes. 
    Extracting the nodes NOT matching the list of tags.
    """

    filtered_dict = defaultdict(list)
    with open(input, "r") as file:
        nodes = json.load(file)
        for elem in nodes["objects"]:
            if type(elem[field]) == list:
                if not any(set(tags).intersection(elem[field])):
                    filtered_dict["objects"].append(elem)
            else:
                if elem[field] not in tags:
                    filtered_dict["objects"].append(elem)

    return filtered_dict


def filterInvalidNodes(in_file, out_file):
    """
    Filtering invalid probes.
    Invalid probes have missing geolocation attributes.
    """

    print(f"Filtering 'invalid' nodes from '{in_file}' to '{out_file}'.")
    filtered = defaultdict(list)

    invalid = 0
    with open(in_file, "r") as input_file:
        nodes = json.load(input_file)
        for elem in nodes["objects"]:

            # longitude and latitude must be valid
            if elem["latitude"] is None or elem["longitude"] is None:
                invalid += 1
                continue

            filtered["objects"].append(elem)

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    print(f"Invalid probes: {invalid} - Correct probes: {len(filtered['objects'])}")


def findConflictingNodes(in_file0, in_file1, out_file):
    """
    Removing conflicting probes from two dictionaries.
    Writing the output to 'out_file'.
    """

    print(f"Searching for 'Conflicting' nodes between '{in_file0}' and '{in_file1}'.")

    with open(in_file0, "r") as input_file0:
        nodes0 = json.load(input_file0)
    with open(in_file1, "r") as input_file1:
        nodes1 = json.load(input_file1)

    filtered_dict = defaultdict(list)

    # find identical nodes in both sets
    for node0 in nodes0["objects"]:
        id0 = node0["id"]
        
        for node1 in nodes1["objects"]:
            id1 = node1["id"]

            if id0 == id1:
                if node0 not in filtered_dict["objects"]:
                    filtered_dict["objects"].append(node0)

    num_conflicting_nodes = len(filtered_dict['objects'])
    if num_conflicting_nodes == 0:
        print(f"Found 0 nodes with conflicting tags.")
    else:
        print(f"Found {num_conflicting_nodes} nodes with conflicting tags - saving them to '{out_file}'.")
        
        with open(out_file, "w") as output_file:
            json.dump(filtered_dict, output_file, indent=4)
    
    return num_conflicting_nodes


def filterConflictingNodes(in_file0, in_file1):
    tmp_file = "conflicting.json"
    num_conflicting_nodes = findConflictingNodes(in_file0, in_file1, tmp_file)

    if num_conflicting_nodes > 0:
        conflicting_ids = list()

        with open(tmp_file, "r") as input_file0:
            nodes0 = json.load(input_file0)

        for node0 in nodes0["objects"]:
            conflicting_ids.append(node0["id"])

        filtered = filterLocalNodesNegated(in_file0, "id", conflicting_ids)
        
        with open(in_file0, "w") as output_file:
            json.dump(filtered, output_file, indent=4)
        
        filtered = filterLocalNodesNegated(in_file1, "id", conflicting_ids)
        
        with open(in_file1, "w") as output_file:
            json.dump(filtered, output_file, indent=4)

    print(f"Removed {num_conflicting_nodes} conflicting nodes from {in_file0} and {in_file1}")



def filterConnected(in_file, out_file):
    """
    Filtering for nodes with status_name 'connected'.
    Writing the resulting json to 'out_file'.
    """

    print(f"Filtering 'Connected and Working' nodes from '{in_file}' to '{out_file}'.")

    # Filter "connected"
    filtered = filterLocalNodes(in_file, "status_name", ["Connected", "connected"])

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    # Filter "IP works"
    filtered = filterLocalNodes(out_file, "tags", ["system-ipv4-works"]) #["system-ipv6-works", "system-ipv4-works"])

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    print(f"Connected probes: {len(filtered['objects'])}")

    # Filtering automatically invalid probes
    filterInvalidNodes(out_file, out_file)


def filterCellular(in_file, out_file):
    print(f"Filtering 'Cellular' nodes from '{in_file}' to '{out_file}'.")

    filtered = filterLocalNodes(in_file, "tags", cellular_tags)
    
    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    print(f"Cellular probes: {len(filtered['objects'])}")


def filterWiFi(in_file, out_file):
    print(f"Filtering 'WiFi' nodes from '{in_file}' to '{out_file}'.")

    filtered = filterLocalNodes(in_file, "tags", wifi_tags)

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)
        
    print(f"WiFi probes: {len(filtered['objects'])}")


def filterSatellite(in_file, out_file):
    print(f"Filtering 'Satellite' nodes from '{in_file}' to '{out_file}'.")

    filtered_v4 = filterLocalNodes(in_file, "asn_v4", [14593])
    filtered_v6 = filterLocalNodes(in_file, "asn_v6", [14593])

    filtered_dict = defaultdict(list)
    for elem in filtered_v4["objects"]:
        filtered_dict["objects"].append(elem)

    for elem in filtered_v6["objects"]:
        if not elem in filtered_dict["objects"]:
            filtered_dict["objects"].append(elem)

    with open(out_file, "w") as output_file:
        json.dump(filtered_dict, output_file, indent=4)

    print(f"Satellite probes: {len(filtered_dict['objects'])}")


def filterLAN(in_file, out_file):
    print(f"Filtering 'LAN' nodes from '{in_file}' to '{out_file}'.")

    filtered = filterLocalNodes(in_file, "tags", ["home"])

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    filtered = filterLocalNodesNegated(out_file, "tags", ["gcs", "aws", "wireless-isp", "isp", "data-center", "datacenter", "datacentre", "satellite", "starlink"] + wifi_tags + cellular_tags)

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    filtered = filterLocalNodesNegated(out_file, "asn_v4", [14593])

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    filtered = filterLocalNodesNegated(out_file, "asn_v6", [14593])

    with open(out_file, "w") as output_file:
        json.dump(filtered, output_file, indent=4)

    print(f"LAN probes: {len(filtered['objects'])}")


# --------------------------------------------------------------------------
# PROBE FILTERING ENDING
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# --------------------------------------------------------------------------


def showAvailableTags(in_file):
    """
    Reading an input file with connected nodes and printing all available user tags which can be filtered for.
    """

    filtered_dict = defaultdict(dict)
    with open(in_file, "r") as input_file:
        nodes = json.load(input_file)
        for node in nodes["objects"]:
            for tag in node["tags"]:
                filtered_dict[tag] = filtered_dict.get(tag, 0) + 1

    for v in sorted(filtered_dict, key=filtered_dict.get, reverse=True):
        print(f"{v}:".ljust(32), f"{filtered_dict[v]}")


def deduplicateIDs(list0, list1):
    """
    Expecting two lists of IDs.
    Returning single list of de-duplicated IDs.
    """

    return list(set(list0).union(list1))

def writeDictToFile(dict, out_file):
    """
    Expecting a dictionary of continent codes to ID list.
    Writing to CSV file format in 'out_file'.
    """

    print(f"Writing dict to '{out_file}'...")

    header = ["Continent", "Elements", "IDs"]

    with open(out_file, "w") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for continent, id_list in dict.items():
            row = [continent, len(id_list), id_list]
            writer.writerow(row)

def saveMatchingIdsToCsv(id_list, lan_id_list, technology, output):
    """
    Expecting two lists of IDs, the access technology and the matched (closest) LAN ID.
    Storing the lists to the given CSV output file.
    """

    print(f"Creating {technology} matching to ({len(id_list)}, {len(lan_id_list)}) and storing to {output}..")

    df = pd.DataFrame(id_list, columns=[f"{technology}"])
    # df[f"{technology}"] = id_list
    df["Lan"] = pd.Series(lan_id_list, dtype=int)
    df.to_csv(output, index=False)

# --------------------------------------------------------------------------
# MATCHING AND MEASUREMENT POINT CREATION
# --------------------------------------------------------------------------


def findDoubleMatches(in_file0, in_file1):
    print(f"Searching for double matches between {in_file0} and {in_file1}")

    with open(in_file0, "r") as input_file0:
        nodes0 = json.load(input_file0)
    with open(in_file1, "r") as input_file1:
        nodes1 = json.load(input_file1)

    meassurement_points = list()

    for node0 in tqdm(nodes0["objects"]):
        loc0 = makeGeolocation(node0["latitude"], node0["longitude"])
        
        # rank all other nodes by distance to node0
        matches_01 = list()

        for node1 in nodes1["objects"]:
            loc1 = makeGeolocation(node1["latitude"], node1["longitude"])
            dist01 = getDistance(loc0, loc1)
            matches_01.append({"id0" : node0["id"], "id1" : node1["id"], "dist" : dist01})
        
        def sortByDist(e):
            return e['dist']

        matches_01.sort(key=sortByDist)

        # take the closests node from each list
        meassurement_points.append({
            'id0' : node0['id'], 
            'id1' : matches_01[0]['id1'], 
            'dist' : matches_01[0]['dist'], 
            })
        
        # We only want the 'best/closest" 100
        meassurement_points.sort(key=sortByDist)
    
    return meassurement_points

def findTripleMatches(in_file0, in_file1, in_file2):
    print(f"Searching for triple matches between {in_file0}, {in_file1} and {in_file2}")

    with open(in_file0, "r") as input_file0:
        nodes0 = json.load(input_file0)
    with open(in_file1, "r") as input_file1:
        nodes1 = json.load(input_file1)
    with open(in_file2, "r") as input_file2:
        nodes2 = json.load(input_file2)

    meassurement_points = list()

    for node0 in tqdm(nodes0["objects"]):
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
    
    return meassurement_points

def combineOnly2Nodes(args):
    with open("probes/cellular.json", "r") as input_cellular:
        cellularProbes = json.load(input_cellular)
    with open("probes/wifi.json", "r") as input_wifi:
        wifiProbes = json.load(input_wifi)
    with open("probes/lan.json", "r") as input_lan:
        lanProbes = json.load(input_lan)

    cellular_ids,lan_ids = findMatchingNodes(cellularProbes, lanProbes)
    filtered_cellular = filterLocalNodes("probes/connected.json", "id", cellular_ids)

    with open("measurement_creation/cellular_test.csv", "w") as output_file:
        output_file.write("Cellular IDs,LAN IDs,Country\n")
        country_codes = getCountryCodes(cellular_ids)
        for i in tqdm(range(len(cellular_ids))):
            output_file.write(str(cellular_ids[i]) + ",")
            output_file.write(str(lan_ids[i]) + ",")
            output_file.write(country_codes[i] + "\n")

    wifi_ids,lan_ids = findMatchingNodes(wifiProbes, lanProbes)
    filtered_cellular = filterLocalNodes("probes/connected.json", "id", wifi_ids)

    with open("measurement_creation/wifi_test.csv", "w") as output_file:
        output_file.write("WiFi IDs,LAN IDs,Country\n")
        country_codes = getCountryCodes(wifi_ids)
        for i in tqdm(range(len(wifi_ids))):
            output_file.write(str(wifi_ids[i]) + ",")
            output_file.write(str(lan_ids[i]) + ",")
            output_file.write(country_codes[i] + "\n")
    
    return

def combineNodes(args):

    with open("probes/cellular.json", "r") as input_nodes_base:
        baseNodes = json.load(input_nodes_base)
    with open("probes/wifi.json", "r") as input_nodes_compare:
        possibleNodes = json.load(input_nodes_compare)

    cellular_ids,wifi_ids = findMatchingNodes(baseNodes, possibleNodes)
    filtered = filterLocalNodes("probes/connected.json", "id", cellular_ids)
    # print(filtered)

    with open("probes/lan.json", "r") as input_nodes_lan:
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

# --------------------------------------------------------------------------
# GEOLOCATION CALCULATIONS
# --------------------------------------------------------------------------


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
    with open("probes/connected.json", "r") as input_file:
        probes = json.load(input_file)
        probes = probes["objects"]
        for id in id_list:
            country_code = next((node["country_code"] for node in probes if node["id"] == id), "Unknown")
            country_codes.append(country_code)

    return country_codes

def sortByCountryCodes(id_list):
    """
    Expecting the probe ID from the RIPE Atlas network.
    Returning the country code.
    """

    country_dict = defaultdict(list)
    with open("probes/connected.json", "r") as input_file:
        probes = json.load(input_file)
        probes = probes["objects"]
        for id in id_list:
            country_code = next((node["country_code"] for node in probes if node["id"] == id), "Unknown")
            
            if country_code in country_dict:
                country_dict[country_code].append(id)
            else:
                country_dict[country_code] = [id]

    return country_dict

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


def getMeasurementNodes(in_file0, in_file1):
    meassurement_points = findDoubleMatches(in_file0, in_file1)

    # print all points
    # counter = 0
    # for elem in meassurement_points:
    #     print(f"[{counter}] {elem}")
    #     counter+=1

    id_list0 = list()
    id_list1 = list()
    for elem in meassurement_points:
        if elem["id0"] not in id_list0:
            id_list0.append(elem["id0"])
        if elem["id1"] not in id_list1:
            id_list1.append(elem["id1"])

    # print(f"# ids from {in_file0} = {len(id_list0)}")
    # print(f"# ids from {in_file1} = {len(id_list1)}")

    return (id_list0, id_list1)

def sortByContinent(country_id_list0, country_id_list1, country_id_list2, country_id_list3):
    continent_dict = defaultdict(list)
    continent_dict["EU"] = []
    continent_dict["NA"] = []
    continent_dict["SA"] = []
    continent_dict["AS"] = []
    continent_dict["AF"] = []
    continent_dict["OC"] = []
    continent_dict["ME"] = []

    eu_countries = ["DE", "BE", "NL", "AT", "ES", "FR", "IT", "GB", "PL", "CZ", "CH", "RU", "FI", "RS", "IL", "SE", "UA", "TR", "AZ", "IE", "DK", "EE", "LV", "SI", "NO", "BG", "GR", "RO", "CY"]
    us_countries = ["US", "CA"]
    sa_countries = ["BR", "VE", "CO"]
    asia_countries = ["SG", "CN", "MY", "JP", "HK", "MN"]
    oc_countries = ["AU", "NZ", "ID"]
    af_countries = ["NA", "AO", "RE", "TN", "ZA", "UG", "BJ"]
    middle_east = ["IN", "IR", "UZ", "PK", "KZ", "IQ", "TJ", "AE", "JO"]


    def match_continent(country, id_list):
        if country in eu_countries:
            continent_dict["EU"] += id_list
        elif country in us_countries:
            continent_dict["NA"] += id_list
        elif country in sa_countries:
            continent_dict["SA"] += id_list
        elif country in asia_countries:
            continent_dict["AS"] += id_list
        elif country in oc_countries:
            continent_dict["OC"] += id_list
        elif country in af_countries:
            continent_dict["AF"] += id_list
        elif country in middle_east:
            continent_dict["ME"] += id_list
        else:
            print(f"[{country}]: ({len(id_list)}) {id_list}")
            continent_dict["Ping"] += id_list

    for country, id_list in country_id_list0.items():
        match_continent(country, id_list)

    for country, id_list in country_id_list1.items():
        match_continent(country, id_list)

    for country, id_list in country_id_list2.items():
        match_continent(country, id_list)

    for country, id_list in country_id_list3.items():
        match_continent(country, id_list)

    return continent_dict


# --------------------------------------------------------------------------
# MEASSUREMENT FUNCTIONS
# --------------------------------------------------------------------------

ping_template = """{
    "definitions": [
        {
            "target": "TARGET_STRING",
            "af": 4,
            "packets": 3,
            "size": 48,
            "description": "CMB Group 9 Part 2",
            "interval": 7200,
            "resolve_on_probe": false,
            "skip_dns_check": false,
            "include_probe_id": false,
            "type": "ping"
        }
    ],
    "probes": [
        {
            "value": "PROBES_CSV_STRING",
            "type": "probes",
            "requested": "PROBES_COUNT"
        }
    ],
    "is_oneoff": false,
    "bill_to": "wagnerc@in.tum.de",
    "stop_time": 1676850900
}
"""

trace_template = """{
    "definitions": [
        {
            "target": "TARGET_STRING",
            "af": 4,
            "response_timeout": 4000,
            "description": "CMB Group 9 Part 2",
            "protocol": "ICMP",
            "interval": 756000,
            "resolve_on_probe": false,
            "packets": 3,
            "size": 48,
            "first_hop": 1,
            "max_hops": 32,
            "paris": 16,
            "destination_option_size": 0,
            "hop_by_hop_option_size": 0,
            "dont_fragment": false,
            "skip_dns_check": false,
            "type": "traceroute"
        }
    ],
    "probes": [
        {
            "value": "PROBES_CSV_STRING",
            "type": "probes",
            "requested": "PROBES_COUNT"
        }
    ],
    "is_oneoff": false,
    "bill_to": "wagnerc@in.tum.de",
    "stop_time": 1676850900
}
"""

curl_template = """curl --dump-header - -H "Content-Type: application/json" -H "Accept: application/json" -X POST -d '{}' https://atlas.ripe.net/api/v2/measurements//?key=f9a735fa-c429-409d-b028-1050a3ee840b"""

def create_measurements(output_file):
    continent_matching = defaultdict(list)
    continent_matching["EU"]=["NA", "AF", "AS", "ME"]
    continent_matching["NA"]=["EU", "SA", "AS", "OC"]
    continent_matching["SA"]=["NA", "AF", "AS", "OC"]
    continent_matching["AF"]=["EU", "SA", "ME"]
    continent_matching["AS"]=["EU", "NA", "SA", "OC"]
    continent_matching["ME"]=["EU", "AF", "OC"]
    continent_matching["OC"]=["NA", "SA", "ME", "AS"]

    # Flash the file content
    open(output_file, "w").close()
    
    output = open(output_file, "w")

    output.write("#!/bin/bash\n\n")
    
    def add_curl_command(source, dest, dest_count, template):
        datacenter = pd.read_csv("measurement_creation/datacenters.csv", sep=",", index_col=None, keep_default_na=False, na_values=["_"], na_filter=False)
        ids = pd.read_csv("measurement_creation/continent_v4.csv", sep=",", index_col=None, keep_default_na=False, na_values=["_"], na_filter=False)

        command = json.loads(template)
        command["definitions"][0]["target"] = list(datacenter[datacenter["Continent"] == dest]["IP"])[dest_count]
        
        for index,row in ids.iterrows():
            if row["Continent"] == source:
                id_list = ast.literal_eval(row["IDs"])
                command["probes"][0]["value"] = str(id_list)[1:-1]
                break

        command["probes"][0]["requested"] = int( ids[ids["Continent"] == source]["Elements"])

        command_config = json.dumps(command)
        curl_command = curl_template.replace("{}", command_config)
        output.write(curl_command + "\n")


    # intra-continental pings
    output.write("# Intra Continental Pings: \n")

    for k,v in tqdm(continent_matching.items()):
        print(f"Source: {k}  Dest: {k}")
        if k == "AF":
            end = 2
        else:
            end = 3
        # print(end)
        for i in range(end):
            add_curl_command(k, k, i, ping_template)

    # inter-continental pings
    output.write("\n\n# Inter Continental Pings: \n")

    for k,v in tqdm(continent_matching.items()):
        for dest in v:
            add_curl_command(k, dest, 0, ping_template)

    # Traceroute
    print("TRACEROUTE")

    output.write("\n\n# Traceroutes: \n")

    # intra-continental traceroutes
    output.write("# Intra Continental Traceroute: \n")
    for k,v in tqdm(continent_matching.items()):
        print(f"Source: {k}  Dest: {k}")
        if k == "AF":
            end = 2
        else:
            end = 3
        # print(end)
        for i in range(end):
            add_curl_command(k, k, i, trace_template)

    # inter-continental traceroutes
    output.write("\n\n# Inter Continental Traceroute: \n")

    for k,v in tqdm(continent_matching.items()):
        for dest in v:
            add_curl_command(k, dest, 0, trace_template)

    output.close()


# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------


if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', type=str, default="20230208.json")
    parser.add_argument('-o', '--output', type=str, default="connected.json")
    parser.add_argument('-t', '--tags', action="store_true", help="Printing all available user-tags and exit.")
    parser.add_argument('-f', '--filter', action="store_true", help="Filtering node types into the given output file. Overwriting existing files!")
    parser.add_argument('-m', '--matching', action="store_true", help="Matching measurement points from existing *.json files. Overwriting existing files!")
    
    args = parser.parse_args()

    if args.tags:
        showAvailableTags(args.input)
        exit(0)

    if args.filter:
        filterConnected(args.input, "probes/connected.json")
        filterCellular("probes/connected.json", "probes/cellular.json")
        filterWiFi("probes/connected.json", "probes/wifi.json")
        filterLAN("probes/connected.json", "probes/lan.json")
        filterSatellite("probes/connected.json", "probes/satellite.json")

        filterConflictingNodes("probes/lan.json", "probes/wifi.json")
        filterConflictingNodes("probes/lan.json", "probes/cellular.json")
        filterConflictingNodes("probes/lan.json", "probes/satellite.json")
        filterConflictingNodes("probes/wifi.json", "probes/cellular.json")
        filterConflictingNodes("probes/wifi.json", "probes/satellite.json")
        filterConflictingNodes("probes/cellular.json", "probes/satellite.json")

    if args.matching:
        # List of IDs that can be used for measurements
        (wifi_ids, wifi_lan_ids) = getMeasurementNodes("probes/wifi.json", "probes/lan.json")
        (cellular_ids, cell_lan_ids) = getMeasurementNodes("probes/cellular.json", "probes/lan.json")
        (satellite_ids, sat_lan_ids) = getMeasurementNodes("probes/satellite.json", "probes/lan.json")

        saveMatchingIdsToCsv(wifi_ids, wifi_lan_ids, "Wifi", "measurement_creation/wifi_lan_match.csv")
        saveMatchingIdsToCsv(cellular_ids, cell_lan_ids, "Cellular", "measurement_creation/cellular_lan_match.csv")
        saveMatchingIdsToCsv(satellite_ids, sat_lan_ids, "Satellite", "measurement_creation/satellite_lan_match.csv")

        lan_ids = deduplicateIDs(cell_lan_ids, wifi_lan_ids)
        lan_ids = deduplicateIDs(lan_ids, sat_lan_ids)

        # TODO: Combine into single function
        country_satellite_ids = sortByCountryCodes(satellite_ids)
        country_wifi_ids = sortByCountryCodes(wifi_ids)
        country_cellular_ids = sortByCountryCodes(cellular_ids)
        country_lan_ids = sortByCountryCodes(lan_ids)

        continent_ids = sortByContinent(country_satellite_ids, country_wifi_ids, country_cellular_ids, country_lan_ids)

        continent_ipv4 = defaultdict(list)
        continent_ipv6 = defaultdict(list)
        with open("probes/connected.json", "r") as input_file:
            probes = json.load(input_file)
            probes = probes["objects"]
            for continent in continent_ids:
                for id in continent_ids[continent]:
                    tags = next((node["tags"] for node in probes if node["id"] == id), [])
                    if "system-ipv4-works" in tags:
                        continent_ipv4[continent].append(id)
                    if "system-ipv6-works" in tags:
                        continent_ipv6[continent].append(id)

        writeDictToFile(continent_ipv4, "measurement_creation/continent_v4.csv")
        writeDictToFile(continent_ipv6, "measurement_creation/continent_v6.csv")

    create_measurements(default_measurement_script)

# --------------------------------------------------------------------------
# END OF MAIN
# --------------------------------------------------------------------------