## 0.
explain probe selection


## 1.
verify probes (wifi really wifi and cellular really cellular)  
 use asn and first hop ips   
check non participating probes  
check offline nodes?  


## 2.
wifi vs lan  
cellular vs lan  
satellite vs lan  
per intra-continent  
=> boxplot per continent difference between access technology and corresponding lan  
check per continent per datacenter diffs  
maybe average diffs per technology?  

### a) Are there significant latency differences between probes using different access technologies i.e. wired vs WiFi vs LTE vs Starlink?
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


over time/day  
=> linegraph (time on x-axis / accumulated diffs on y-axis) one line per technology  
one graph per continent  

{ ping measurement use verage of 3 packets }  
{ packetloss is still a valid measurement }  
(compare overall packetlos with internet packetloss)  
(per intra-continent per datacenter)  


## 3.
### 3 a)
check intra-continental asn numbers in traces from nodes to datacenters  
=> pie chart of aggregated asns found  

check intra-continental as numbers per datacenters  

check inter-continetal latencies  

### 3 b)
try to plot cdf of inter-continental(selective) trace latencies  
 per technology  
compare atlantic/pacific ocean vs EU<->ME etc..  

### 3 c)
argue with necessary end-to-end latency  
argue with wifi/cellular/satellite added latency  
use X-hop trace latency to imaging edge system  

i) same country  
ii) same continent different country  
iii) different continent  

