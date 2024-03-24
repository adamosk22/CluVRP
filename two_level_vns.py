import xml.etree.ElementTree as ET
import numpy as np
import sys
import math
import random

def extract_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    nodes = root.findall('NodeP')

    data = []
    for node in nodes:
        node_data = {}
        node_data['Addr'] = node.find('Addr').text
        node_data['id'] = int(node.find('id').text)
        node_data['Cluster'] = int(node.find('Cluster').text)
        node_data['DemDelivery'] = int(node.find('DemEnt').text)
        node_data['DemPickup'] = int(node.find('DemRec').text)
        node_data['CoordX'] = int(node.find('CoordX').text)
        node_data['CoordY'] = int(node.find('CoordY').text)
        data.append(node_data)

    return data

def create_clusters_table(nodes):
    clusters = {}  # Słownik przechowujący klastry i ich łączny ładunek

    # Iteruj przez wierzchołki i zsumuj ładunki dla każdego klastra
    for node in nodes:
        cluster_id = node['Cluster']
        demand_delivery = node['DemDelivery']
        demand_pickup = node['DemPickup']
        total_demand = demand_delivery + demand_pickup

        if cluster_id in clusters:
            clusters[cluster_id]['TotalDemand'] += total_demand
        else:
            if cluster_id!=0:
                clusters[cluster_id] = {'TotalDemand': total_demand}

    # Tworzenie tabeli z wynikami
    table = []
    for cluster_id, cluster_data in clusters.items():
        row = {'ClusterID': cluster_id, 'TotalDemand': cluster_data['TotalDemand']}
        table.append(row)

    return table

    

def euclidean_distance(point1, point2):
    x1, y1 = point1['CoordX'], point1['CoordY']
    x2, y2 = point2['CoordX'], point2['CoordY']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_inter_cluster_distances(data):
    n_clusters = max(node['Cluster'] for node in data) + 1
    cluster_coords = {cluster_id: [] for cluster_id in range(n_clusters)}

    # Zbieranie współrzędnych punktów dla każdego klastra
    for node in data:
        cluster_coords[node['Cluster']].append({'CoordX': node['CoordX'], 'CoordY': node['CoordY']})

    # Obliczanie najkrótszych odległości między klastrami
    inter_cluster_distances = [[math.inf] * n_clusters for _ in range(n_clusters)]
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            min_dist = min(euclidean_distance(p1, p2) for p1 in cluster_coords[i] for p2 in cluster_coords[j])
            inter_cluster_distances[i][j] = min_dist
            inter_cluster_distances[j][i] = min_dist
    
    return inter_cluster_distances

def allocate_clusters_to_vehicle(clusters_data, vehicle_capacity):
    clusters_sorted = sorted(clusters_data, key=lambda x: x['TotalDemand'], reverse=True)
    vehicles= [{'clusters': [], 'remaining_capacity': vehicle_capacity}]

    for cluster in clusters_sorted:
        cluster_demand = cluster['TotalDemand']
        allocated = False

        for vehicle in vehicles:
            if vehicle['remaining_capacity'] >= cluster_demand:
                vehicle['clusters'].append(cluster['ClusterID'])
                vehicle['remaining_capacity'] -= cluster_demand
                allocated = True
                break

        if not allocated:
            new_vehicle = {'clusters': [cluster['ClusterID']], 'remaining_capacity': vehicle_capacity - cluster_demand}
            vehicles.append(new_vehicle)

    return vehicles

def calculate_route_cost(clusters_data, inter_cluster_distances):
    # Przykładowa funkcja obliczająca koszt trasy na podstawie danych klastrów
    total_cost = 0
    for i in range(len(clusters_data) - 1):
        print('test')
        print(clusters_data)
        clustersDict = clusters_data[1]
        print(type(clustersDict))
        print(clustersDict)
        clusters = clustersDict['clusters']
        cluster1 = clusters[i]
        cluster2 = clusters[i + 1]
        total_cost += inter_cluster_distances[cluster1][cluster2]
        print(total_cost)
    return total_cost

def intra_swap(clusters_data):
    # Operator Swap: zamienia pozycje dwóch klastrów w trasie
    if len(clusters_data) >= 2:
        idx1, idx2 = random.sample(range(len(clusters_data)), 2)
        clusters_data[idx1], clusters_data[idx2] = clusters_data[idx2], clusters_data[idx1]

def intra_relocate(clusters_data):
    # Operator Relocate: usuwa jeden klaster i wstawia go w inne miejsce w trasie
    if len(clusters_data) >= 2:
        cluster = random.choice(clusters_data)
        clusters_data.remove(cluster)
        idx = random.randint(0, len(clusters_data))
        clusters_data.insert(idx, cluster)

def generate_random_order(neighbourhoods):
    # Generuj losową kolejność sprawdzania sąsiedztw
    random_order = random.sample(neighbourhoods, len(neighbourhoods))
    return random_order

def vns_cluster_level(clusters, max_iterations, inter_cluster_distances):

    # Ustalenie sąsiedztw i operacji
    neighbourhoods = [intra_swap, intra_relocate]
    best_clusters = clusters
    best_cost = calculate_route_cost(clusters, inter_cluster_distances)
    nIterationsNoImprovement = 0

    # Główna pętla VNS
    for _ in range(max_iterations):
        random_order = generate_random_order(neighbourhoods)
        for neighbourhood in random_order:
            neighbourhood(clusters)
            total_cost = calculate_route_cost(clusters, inter_cluster_distances)
            if total_cost < best_cost:
                best_clusters = clusters.copy()
                best_cost = total_cost
                nIterationsNoImprovement = 0
            else:
                nIterationsNoImprovement += 1
        
        if nIterationsNoImprovement >= max_iterations // 2:
            break
    
    return best_clusters, best_cost

# Stałe i zmienne
nIterationsNoImprovement = 0
goToNodeVNS = False
stoppingCriterion = False
maxIterationsNoImprovement = 100
cluVNSProb = 0.5
S_star_i = None
f_S_star_i = float('inf')
vehicle_capacity = 500
counter = 0

# Step 0: Precomputation
xml_file_path = 'dataset.xml'

nodes = extract_data_from_xml(xml_file_path)
inter_cluster_distances = calculate_inter_cluster_distances(nodes)
for row in inter_cluster_distances:
    print(row)

# Step 1: Constructive phase
clusters = create_clusters_table(nodes)
for cluster in clusters:
    print(cluster)
allocated_clusters = allocate_clusters_to_vehicle(clusters, vehicle_capacity)
for i, vehicle in enumerate(allocated_clusters, 1):
    print(f"Vehicle {i}: {vehicle['clusters']} (Remaining Capacity: {vehicle['remaining_capacity']})")
    #initial_cost = calculate_route_cost(vehicle['clusters'], inter_cluster_distances)
    #print("Initial Cost:", initial_cost)

# Step 2: Intensification phase
while counter==0:
    counter += 1
    for vehicle in enumerate(allocated_clusters, 1):
        best_clusters, best_cost = vns_cluster_level(vehicle, 100, inter_cluster_distances)
        for cluster in best_clusters:
            print(f"ClusterID: {cluster['ClusterID']}, Total Demand: {cluster['TotalDemand']}")
        print("Best Cost:", best_cost)


