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

    for vehicle in vehicles:
        vehicle['clusters'].append(0)
        vehicle['clusters'].insert(0,0)

    return vehicles

def calculate_route_cost(clusters_data, inter_cluster_distances):
    # Funkcja obliczająca koszt trasy na podstawie danych klastrów
    total_cost = 0
    for i in range(len(clusters_data) - 1):
        cluster1 = clusters_data[i]
        cluster2 = clusters_data[i + 1]
        total_cost += inter_cluster_distances[cluster1][cluster2]
    return total_cost

def intra_swap(allocated_clusters):
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        clusters_data = clusters_data[1:-1]
        # Operator Swap: zamienia pozycje dwóch klastrów w trasie
        if len(clusters_data) >= 2:
            idx1, idx2 = random.sample(range(len(clusters_data)), 2)
            clusters_data[idx1], clusters_data[idx2] = clusters_data[idx2], clusters_data[idx1]
        clusters_data.append(0)
        clusters_data.insert(0,0)
        vehicle['clusters'] = clusters_data
    

def intra_relocate(allocated_clusters):
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        clusters_data = clusters_data[1:-1]
        # Operator Relocate: usuwa jeden klaster i wstawia go w inne miejsce w trasie
        if len(clusters_data) >= 2:
            cluster = random.choice(clusters_data)
            clusters_data.remove(cluster)
            idx = random.randint(0, len(clusters_data))
            clusters_data.insert(idx, cluster)
        clusters_data.append(0)
        clusters_data.insert(0,0)
        vehicle['clusters'] = clusters_data

def intra_two_opt(allocated_clusters):
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        clusters_data = clusters_data[1:-1]
        # Operator Two-Opt: zamienia kolejność dwóch krawędzi w trasie
        if len(clusters_data) >= 4:
            idx1, idx2 = random.sample(range(len(clusters_data)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            clusters_data[idx1:idx2+1] = reversed(clusters_data[idx1:idx2+1])
        clusters_data.append(0)
        clusters_data.insert(0,0)
        vehicle['clusters'] = clusters_data

def intra_or_opt(allocated_clusters):
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        clusters_data = clusters_data[1:-1]
        # Operator Or-Opt: usuwa N kolejnych klastrów i wstawia je w inne miejsce w trasie (N=2, 3, 4)
        if len(clusters_data) >= 4:
            N = random.choice([2, 3, 4])
            idx = random.randint(0, len(clusters_data) - N)
            clustersChosen = clusters_data[idx:idx+N]
            clusters_data[idx:idx+N] = []
            new_idx = random.randint(0, len(clusters_data))
            clusters_data[new_idx:new_idx] = clustersChosen
        clusters_data.append(0)
        clusters_data.insert(0,0)
        vehicle['clusters'] = clusters_data

# Operatory lokalnego przeszukiwania (inter vehicle)
def inter_swap(vehicles_routes):
    for vehicle in allocated_clusters:
        vehicle['clusters'] = vehicle['clusters'][1:-1]
    # Operator Swap: zamienia pojazd dla dwóch klastrów
    if len(vehicles_routes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles_routes)), 2)
        cluster1 = random.choice(vehicles_routes[idx1]['clusters'])
        cluster2 = random.choice(vehicles_routes[idx2]['clusters'])
        demand1 = clusters[cluster1 - 1]['TotalDemand']
        demand2 = clusters[cluster2 - 1]['TotalDemand']
        capacity1 = vehicles_routes[idx1]['remaining_capacity'] + demand1 - demand2
        capacity2 = vehicles_routes[idx2]['remaining_capacity'] + demand2 - demand1
        if (capacity1 > 0) and (capacity2 > 0):
            vehicles_routes[idx1]['clusters'].remove(cluster1)
            vehicles_routes[idx2]['clusters'].remove(cluster2)
            vehicles_routes[idx1]['clusters'].append(cluster2)
            vehicles_routes[idx2]['clusters'].append(cluster1)
            vehicles_routes[idx1]['remaining_capacity'] = vehicles_routes[idx1]['remaining_capacity'] + demand1 - demand2
            vehicles_routes[idx2]['remaining_capacity'] = vehicles_routes[idx2]['remaining_capacity'] - demand1 + demand2
    for vehicle in allocated_clusters:
        vehicle['clusters'].append(0)
        vehicle['clusters'].insert(0,0)

def inter_relocate(vehicles_routes):
    for vehicle in allocated_clusters:
        vehicle['clusters'] = vehicle['clusters'][1:-1]
    # Operator Relocate: usuwa klaster z jednego pojazdu i wstawia go do innego pojazdu
    if len(vehicles_routes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles_routes)), 2)
        cluster = random.choice(vehicles_routes[idx1]['clusters'])
        demand = clusters[cluster - 1]['TotalDemand']
        capacity = vehicles_routes[idx2]['remaining_capacity'] - demand
        if capacity > 0:
            vehicles_routes[idx1]['clusters'].remove(cluster)
            vehicles_routes[idx2]['clusters'].append(cluster)
            vehicles_routes[idx1]['remaining_capacity'] = vehicles_routes[idx1]['remaining_capacity'] + demand
            vehicles_routes[idx2]['remaining_capacity'] = vehicles_routes[idx2]['remaining_capacity'] - demand
    for vehicle in allocated_clusters:
        vehicle['clusters'].append(0)
        vehicle['clusters'].insert(0,0)

def inter_or_opt(vehicles_routes):
    for vehicle in allocated_clusters:
        vehicle['clusters'] = vehicle['clusters'][1:-1]
    # Operator Or-Opt: usuwa N kolejnych klastrów z jednego pojazdu i wstawia je do innego pojazdu (N=2, 3, 4)
    if len(vehicles_routes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles_routes)), 2)
        N = random.choice([2, 3, 4])
        if N > len(vehicles_routes[idx1]['clusters']):
            N = len(vehicles_routes[idx1]['clusters'])
        chosenClusters = random.sample(vehicles_routes[idx1]['clusters'], N)
        demand = 0
        for cluster in chosenClusters:
            demand += clusters[cluster - 1]['TotalDemand']
        capacity = vehicles_routes[idx2]['remaining_capacity'] - demand
        if capacity > 0:
            for cluster in chosenClusters:
                vehicles_routes[idx1]['clusters'].remove(cluster)
                vehicles_routes[idx2]['clusters'].append(cluster)
                vehicles_routes[idx1]['remaining_capacity'] = vehicles_routes[idx1]['remaining_capacity'] + demand
                vehicles_routes[idx2]['remaining_capacity'] = vehicles_routes[idx2]['remaining_capacity'] - demand
    for vehicle in allocated_clusters:
        vehicle['clusters'].append(0)
        vehicle['clusters'].insert(0,0)

def generate_random_order(neighbourhoods):
    # Generuj losową kolejność sprawdzania sąsiedztw
    random_order = random.sample(neighbourhoods, len(neighbourhoods))
    return random_order

def vns_cluster_level(allocated_clusters, max_iterations, inter_cluster_distances):
    # Ustalenie sąsiedztw i operacji
    neighbourhoods = [intra_swap, intra_relocate, intra_two_opt, intra_or_opt, inter_swap, inter_relocate, inter_or_opt]
    best_clusters = clusters
    best_costs = []
    for vehicle in allocated_clusters:
        best_costs.append(calculate_route_cost(vehicle['clusters'], inter_cluster_distances))
    best_cost = max(best_costs)
    nIterationsNoImprovement = 0

    # Główna pętla VNS
    for _ in range(max_iterations):
        random_order = generate_random_order(neighbourhoods)
        for neighbourhood in random_order:
            neighbourhood(allocated_clusters)
            total_costs = []
            for vehicle in allocated_clusters:
                total_costs.append(calculate_route_cost(vehicle['clusters'], inter_cluster_distances))
            if max(total_costs) < best_cost:
                best_clusters = allocated_clusters.copy()
                best_costs = total_costs
                best_cost = max(best_costs)
                nIterationsNoImprovement = 0
            else:
                nIterationsNoImprovement += 1
        
        if nIterationsNoImprovement >= max_iterations // 2:
            break
    
    return best_clusters, best_costs


# Stałe i zmienne
nIterationsNoImprovement = 0
goToNodeVNS = False
stoppingCriterion = False
maxIterationsNoImprovement = 100
cluVNSProb = 0.5
best_clusters = None
best_cost = float('inf')
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
while (counter==0) or (stoppingCriterion == False):
    counter += 1
    calculated_clusters_list = list()
    calcualted_cost_list = list()
    calculated_clusters, calculated_cost = vns_cluster_level(allocated_clusters, 100, inter_cluster_distances)
    i = 1
    for vehicle in calculated_clusters:
        print("Vehicle:", i, "Clusters order:", vehicle)
        i += 1
    print("Calculated Cost:", calculated_cost)
    if max(calculated_cost)<best_cost:
            best_cost = max(calculated_cost)
            nIterationsNoImprovement=0
    else:
        nIterationsNoImprovement += 1
        print("No improvement in ", nIterationsNoImprovement, "iterations")
    print("Best cost:", best_cost)
    if nIterationsNoImprovement==maxIterationsNoImprovement:
        stoppingCriterion=True
        



