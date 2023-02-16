## 0.
explain probe selection


## 1.
verify probes (wifi really wifi and cellular really cellular)  
 use asn and first hop ips   
check non participating probes  
check offline nodes?  


## 2.

### 2 a) Are there significant latency differences between probes using different access technologies i.e. wired vs WiFi vs LTE vs Starlink?
- generate node matchings (and save them to a file or smth)
    - wifi_ids, wifi_lan_ids
    - cellular_ids, cellular_lan_ids
    - satellite_ids, satellite_lan_ids
- for each wifi_id  
  for each continent  
  filter all meassurements to datacenter1 from csv  
  aggregate all meassurements into one avg  
- for each wifi_lan_id  
  for each continent  
  filter all meassurements to datacenter1 from csv  
  aggregate all meassurements into one avg  
- for each pair of wifi node and lan node  
  take difference = wifi_latency - wifi_lan_latency  
- plot boxplot of differences (per continent)
- repeat for datacenter2 and datacenter3 (include in boxplot)
- repeat for cellular and satellite nodes  
  one boxplot per technology

{ ping measurement use average of 3 packets }  
{ packetloss is still a valid measurement }  
(compare overall packetlos with internet packetloss)  
(per intra-continent per datacenter)  

### How does the latencies vary over time/day?
- Same procedure as above, but
  - only one datacenter
  - aggregate meassurements per hour (average per hour)
  - plot as linegraph (one line per continent)  
    one plot per technology

### 2 b) Compare the state of different last-mile access technology in different continents.  Is there an obviously better-performing wireless technology in every continent? Which and by how much?
- Analyse plots from 2a)


## 3. What is the impact of the distance between the probe and cloud datacenter to end-to-end latency?

### 3 a) Which organizations host the routers on-path between probes and cloud datacenters?
- get intra-continental asn numbers in traces from nodes to datacenters
  - use all nodes in a continent
  - aggregate all asns (+ count how often)
  - per datacenter (only intra-continent)
- plot as pie chart (one per continent)

### Do you observe different Internet routes for different cloud providers? If yes, why?
- check differences between datacenters  

### 3 b) What are the latency variations for probes connecting to datacenters in neighboring continents via under-sea cables
- Pick some neighbors (e.g. EU<->US, EU<->ME, ...)
- use all LAN probes in a continent
- target same datacenter
- (aggregate node latencies? how? or just pick some?)
- plot cdf of inter-continental trace latencies

### Do you observe similar variations for wired and satellite probes? Why/why not?
- repeat, but
- use wifi, satellite nodes
- plot one cdf of both (or maybe two lines in the same plot if easier) 

### 3 c) If you imagine the wireless RIPE Atlas probes to be mobile clients running a next-generation application requiring 30 ms end-to-end operational latency, is there a potential benefit in offloading to an organization on path instead of offloading to the cloud datacenter deployed argue with necessary end-to-end latency argue with wifi/cellular/satellite added latency use X-hop trace latency to imaging edge system  

#### i) same country
- plot cdf with nodes only in country (germany)
- argue with intra-country cdf
#### ii) same continent different country
- argue with cross-country cdf (france->germany?)
#### iii) different continent
- argue with inter-continent cdf (reuse US<->EU cdf from 3b) )

