import requests
import json
import posixpath
from argparse import ArgumentParser
from urllib.parse import urljoin

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

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for throughput values from pcap file')
    parser.add_argument('-i', '--input', action="append", default=[], required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    

    filterWifiNodes()