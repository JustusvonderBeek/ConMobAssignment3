import pandas as pd
import socket
from argparse import ArgumentParser

# --------------------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------------------

continents = 7
datacenters = 3
other_continents = 4
interval_per_day = 12
days = 10
ping_cost = 3
traceroute_cost = 60

eu = 348
na = 117
sa = 8
_as = 22
af = 8
oc = 26
me =18

# --------------------------------------------------------------------------
# CALCULATION
# --------------------------------------------------------------------------

def calculatePing():
    total_probes = eu + na + sa + _as + af + oc + me
    intra_cont = total_probes * interval_per_day * days * datacenters * ping_cost
    inter_cont = total_probes * other_continents * interval_per_day * days * ping_cost

    total = inter_cont + intra_cont

    print(f"Total probes: {total_probes}")
    print("PING:")
    print(f"Intra cost: {intra_cont}")
    print(f"Inter cost: {inter_cont}")
    print(f"Total: {total}")

    return total

def calculateTraceroute():
    total_probes = eu + na + sa + _as + af + oc + me
    intra_cont = total_probes * datacenters * traceroute_cost 
    inter_cont = total_probes * other_continents * datacenters * traceroute_cost
    total = inter_cont + intra_cont

    print("TRACEROUTE:")
    print(f"Intra cost: {intra_cont}")
    print(f"Inter cost: {inter_cont}")
    print(f"Total: {total}")

    return total

def fillIPs(args):

    data = pd.read_csv(args.input, delimiter=",")
    # print(data.to_markdown())

    for index,row in data.iterrows():
        # print(row["Datacenter"])
        ip = socket.gethostbyname(row["Datacenter"])
        # print(ip)
        elem = data.iloc[index]
        elem["IP"] = ip
        # print(elem)
        data.at[index] = elem

    print(data.to_markdown())

    data.to_csv(args.input, sep=",", index=False)

# --------------------------------------------------------------------------
# MAIN METHOD
# --------------------------------------------------------------------------

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate CDF plots to compare the ping latency')
    parser.add_argument('-i','--input', type=str, default="datacenters.csv", help="The path to the CSV file containing the datacenter information.")
    parser.add_argument('-c','--cost', action="store_true", help="If set, the costs are calculated.")

    args = parser.parse_args()

    if args.cost:
        ping = calculatePing()
        traceroute = calculateTraceroute()

        print(f"Total credits: {ping + traceroute}")

        exit(0)

    fillIPs(args)