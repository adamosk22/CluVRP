import xml.etree.ElementTree as ET
import numpy as np
import sys
import math
import random
import networkx as nx
import time

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

def solve_tsp_in_cluster(cluster_data):
    # Tworzenie grafu dla klientów w klastrze
    G = nx.Graph()
    for customer in cluster_data:
        G.add_node(customer['id'], pos=(customer['CoordX'], customer['CoordY']))
    
    # Dodanie krawędzi między klientami w grafie na podstawie odległości (np. odległość euklidesowa)
    for i in range(len(cluster_data)):
        for j in range(i+1, len(cluster_data)):
            dist = ((cluster_data[i]['CoordX'] - cluster_data[j]['CoordX']) ** 2 + 
                    (cluster_data[i]['CoordY'] - cluster_data[j]['CoordY']) ** 2) ** 0.5
            G.add_edge(cluster_data[i]['id'], cluster_data[j]['id'], weight=dist)

    # Rozwiązanie problemu TSP jako cyklu Hamiltona
    tsp_path = nx.approximation.traveling_salesman_problem(G)
    return tsp_path

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

def create_nodes_clusters_table(nodes):
    cluster_nodes = {}  # Słownik do przechowywania danych po przekształceniu

    for item in nodes:
        cluster_id = item['Cluster']
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []  # Inicjalizacja listy dla klastra, jeśli jeszcze nie istnieje
        cluster_nodes[cluster_id].append({'id': item['id'], 'CoordX': item['CoordX'], 'CoordY': item['CoordY']})
    
    return cluster_nodes

    

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

def calculate_route_cost_clusters(clusters_data, inter_cluster_distances):
    # Funkcja obliczająca koszt trasy na podstawie danych klastrów
    total_cost = 0
    for i in range(len(clusters_data) - 1):
        cluster1 = clusters_data[i]
        cluster2 = clusters_data[i + 1]
        total_cost += inter_cluster_distances[cluster1][cluster2]
    return total_cost

def calculate_root_cost_nodes(customers_data):
    total_cost = 0
    for i in range(len(customers_data) - 1):
        node1 = customers_data[i]
        node2 = customers_data[i+1]
        total_cost += euclidean_distance(node1, node2)
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

def intra_swap_nodes(allocated_nodes):
    for vehicle in allocated_nodes:
        current_cluster = 0
        current_nodes = []
        i = 0
        for node in vehicle:
            if node['Cluster'] == current_cluster:
                current_nodes.append(i)
            else:
                # Operator Swap: zamienia pozycje dwóch klastrów w trasie
                if len(current_nodes) >=2:
                    idx1, idx2 = random.sample((current_nodes), 2)
                    vehicle[idx1], vehicle[idx2] = vehicle[idx2], vehicle[idx1]
                current_nodes.clear()
                current_nodes.append(i)
                current_cluster = node['Cluster']
            i += 1

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

def intra_relocate_nodes(allocated_nodes):
    for vehicle in allocated_nodes:
        current_cluster = 0
        current_nodes = []
        i = 0
        for node in vehicle:
            if node['Cluster'] == current_cluster:
                current_nodes.append(i)
            else:
                # Operator Relocate: usuwa jeden wierzchołek i wstawia go w inne miejsce w trasie
                if len(current_nodes) >=2:
                    nodeId = random.choice(current_nodes)
                    node = vehicle[nodeId].copy()
                    del vehicle[nodeId]
                    idx = 0
                    while vehicle[idx]['Cluster'] != current_cluster:
                        idx = random.randint(1,len(vehicle)-2)
                    vehicle.insert(idx, node)
                current_nodes.clear()
                current_nodes.append(i)
                current_cluster = node['Cluster']
            i += 1

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

def intra_two_opt_nodes(allocated_nodes):
    for vehicle in allocated_nodes:
        current_cluster = 0
        current_nodes = []
        i = 0
        for node in vehicle:
            if node['Cluster'] == current_cluster:
                current_nodes.append(i)
            else:
                # Operator Two-Opt: zamienia kolejność dwóch krawędzi w trasie
                if len(current_nodes) >= 4:
                    idx1, idx2 = random.sample(current_nodes, 2)
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    vehicle[idx1:idx2+1] = reversed(vehicle[idx1:idx2+1])
                current_nodes.clear()
                current_nodes.append(i)
                current_cluster = node['Cluster']
            i += 1

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

def intra_or_opt_nodes(allocated_nodes):
    for vehicle in allocated_nodes:
        current_cluster = 0
        current_nodes = []
        i = 0
        for node in vehicle:
            if node['Cluster'] == current_cluster:
                current_nodes.append(i)
            else:
                # Operator Or-Opt: usuwa N kolejnych węzłów i wstawia je w inne miejsce w trasie (N=2, 3, 4)
                if len(current_nodes) >= 4:
                    N = random.choice([2, 3, 4])
                    idx = random.choice(current_nodes)
                    while (idx+N) > len(current_nodes):
                        N -= 1
                    if N >= 2:
                        nodesChosen = vehicle[idx:idx+N].copy()
                        N_temp = N
                        for i in range(idx,idx+N_temp):
                            if vehicle[i]['Cluster'] != current_cluster:
                                nodesChosen.remove(vehicle[i])
                                N =- 1
                        vehicle[idx:idx+N] = []
                        new_idx = 0
                        while vehicle[new_idx]['Cluster'] != current_cluster:
                            new_idx = random.randint(1,len(vehicle)-2)
                        vehicle[new_idx:new_idx] = nodesChosen
                current_nodes.clear()
                current_nodes.append(i)
                current_cluster = node['Cluster']
            i += 1

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

def inter_swap_nodes(allocated_nodes):
    vehicle_number = 0
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        # Operator Swap: zamienia pozycje dwóch klastrów w trasie
        if len(clusters_data) >= 2:
            idx1, idx2 = random.sample(range(len(clusters_data)), 2)
            vehicle_nodes = allocated_nodes[vehicle_number]
            indices1 = [idx for idx, elem in enumerate(vehicle_nodes) if elem['Cluster'] == clusters_data[idx1]]
            indices2 = [idx for idx, elem in enumerate(vehicle_nodes) if elem['Cluster'] == clusters_data[idx2]]
            i = 0
            if len(indices1) == len(indices2):
                for i, indice in enumerate(indices1):
                    vehicle_nodes[indices1[i]], vehicle_nodes[indices2[i]] = vehicle_nodes[indices2[i]], vehicle_nodes[indices1[i]]
        vehicle_number += 1

def inter_swap_nodes_vehicle(allocated_nodes):
    # Operator Swap: zamienia pojazd dla dwóch klastrów
    if len(allocated_nodes) >= 2:
        vehicle_number1, vehicle_number2 = random.sample(range(len(allocated_clusters)), 2)
        cluster1 = random.choice(allocated_clusters[vehicle_number1]['clusters'])
        cluster2 = random.choice(allocated_clusters[vehicle_number2]['clusters'])
        demand1 = clusters[cluster1 - 1]['TotalDemand']
        demand2 = clusters[cluster2 - 1]['TotalDemand']
        capacity1 = allocated_clusters[vehicle_number1]['remaining_capacity'] + demand1 - demand2
        capacity2 = allocated_clusters[vehicle_number2]['remaining_capacity'] + demand2 - demand1
        if (capacity1 > 0) and (capacity2 > 0):
            vehicle_nodes1 = allocated_nodes[vehicle_number1]
            vehicle_nodes2 = allocated_nodes[vehicle_number2]
            indices1 = [idx for idx, elem in enumerate(vehicle_nodes1) if elem['Cluster'] == cluster1]
            indices2 = [idx for idx, elem in enumerate(vehicle_nodes2) if elem['Cluster'] == cluster2]
            if len(indices1) == len(indices2):
                for i, indice in enumerate(indices1):
                    vehicle_nodes1[indices1[i]], vehicle_nodes2[indices2[i]] = vehicle_nodes2[indices2[i]], vehicle_nodes1[indices1[i]]

    
        
    

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

def inter_relocate_nodes(allocated_nodes):
    vehicle_number = 0
    for vehicle in allocated_clusters:
        clusters_data = vehicle['clusters']
        # Operator Relocate: usuwa jeden klaster i wstawia go w inne miejsce w trasie
        if len(clusters_data) >= 2:
            cluster = random.choice(clusters_data)
            if(cluster != 0):
                idx = random.randint(1, len(allocated_nodes[vehicle_number]) - 2)
                if allocated_nodes[vehicle_number][0]!=allocated_nodes[vehicle_number][-1]:
                    allocated_nodes[vehicle_number][-1]=allocated_nodes[vehicle_number][0]
                while (allocated_nodes[vehicle_number][idx]['Cluster'] == allocated_nodes[vehicle_number][idx+1]['Cluster']) and (idx<len(allocated_nodes[vehicle_number])-1):
                    idx += 1
                indices1 = [idx for idx, elem in enumerate(allocated_nodes[vehicle_number]) if elem['Cluster'] == cluster]
                relocated_elements = []
                for indice in sorted(indices1, reverse=True):
                    relocated_elements.append(allocated_nodes[vehicle_number].pop(indice))
                for element in relocated_elements:
                    allocated_nodes[vehicle_number].insert(idx+1, element)
        vehicle_number += 1

def inter_relocate_nodes_vehicle(allocated_nodes):
    # Operator Relocate: usuwa klaster z jednego pojazdu i wstawia go do innego pojazdu
    if len(allocated_clusters) >= 2:
        idx1, idx2 = random.sample(range(len(allocated_clusters)), 2)
        cluster = random.choice(allocated_clusters[idx1]['clusters'])
        demand = clusters[cluster - 1]['TotalDemand']
        capacity = allocated_clusters[idx2]['remaining_capacity'] - demand
        if capacity > 0 and cluster != 0:
            idx = random.randint(1, len(allocated_nodes[idx2]) - 2)
            if(allocated_nodes[idx2][0]!=allocated_nodes[idx2][-1]):
                allocated_nodes[idx2].append(allocated_nodes[idx2][0])
            while allocated_nodes[idx2][idx]["Cluster"] == allocated_nodes[idx2][idx+1]['Cluster']:
                if idx < len(allocated_nodes[idx2]) - 2:
                    idx += 1
            indices1 = [idx for idx, elem in enumerate(allocated_nodes[idx1]) if elem['Cluster'] == cluster]
            relocated_elements = []
            for indice in sorted(indices1, reverse=True):
                relocated_elements.append(allocated_nodes[idx1].pop(indice))
            for element in relocated_elements:
                allocated_nodes[idx2].insert(idx+1, element)

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

def vns_cluster_level(allocated_clusters, max_iterations, inter_cluster_distances, best_clusters):
    # Ustalenie sąsiedztw i operacji
    neighbourhoods = [intra_swap, intra_relocate, intra_two_opt, intra_or_opt, inter_swap, inter_relocate, inter_or_opt]
    best_costs = []
    for vehicle in allocated_clusters:
        best_costs.append(calculate_route_cost_clusters(vehicle['clusters'], inter_cluster_distances))
    best_cost = max(best_costs)

    # Główna pętla VNS
    random_order = generate_random_order(neighbourhoods)
    optimum_reached = False
    while optimum_reached == False:
        for i, neighbourhood in enumerate(random_order):
            neighbourhood(allocated_clusters)
            total_costs = []
            for vehicle in allocated_clusters:
                total_costs.append(calculate_route_cost_clusters(vehicle['clusters'], inter_cluster_distances))
                if max(total_costs) < best_cost:
                    if len(allocated_clusters) == len(best_clusters):
                        if(allocated_clusters != []):
                            best_clusters = allocated_clusters.copy()
                        best_costs = total_costs
                        best_cost = max(best_costs)
                    else:
                        if(allocated_clusters != []):
                            best_clusters = allocated_clusters.copy()
                    break
                else:
                    if i == len(random_order) - 1:
                        optimum_reached = True
                    best_clusters = allocated_clusters.copy()
    
    return best_clusters, best_costs

def convert_cluster_to_customer_sequence(nodes, cluster_tsp_paths, calculated_clusters, rand_conversion_prob):
    converted_sequence = []
    node0 = nodes[cluster_tsp_paths[0][0]]
    last_node = node0
    for vehicle in calculated_clusters:
        vehicle_sequence = []
        for cluster in vehicle['clusters']:
            cluster_nodes = []
            for node in nodes:
                if node['Cluster'] == cluster:
                    cluster_nodes.append(node)
            cluster_nodes.sort(key = lambda x: euclidean_distance(x, last_node))
            starter_node = cluster_nodes[0]
            vehicle_sequence.append(starter_node)
            if random.random() < rand_conversion_prob:
                random.shuffle(cluster_nodes)  # Randomize the order of nodes
            last_node = cluster_nodes[-1]
            for cluster_node in cluster_nodes:
                if cluster_node not in vehicle_sequence:
                    vehicle_sequence.append(cluster_node)
        converted_sequence.append(vehicle_sequence)
    return converted_sequence

def vns_customer_level(customer_sequence):
    best_nodes_cost = []
    best_nodes = customer_sequence
    neighbourhoods = [intra_or_opt_nodes, intra_two_opt_nodes, intra_relocate_nodes, intra_swap_nodes, inter_swap_nodes, inter_relocate_nodes, inter_swap_nodes_vehicle, inter_relocate_nodes_vehicle]
    for vehicle in customer_sequence:
        best_nodes_cost.append(calculate_root_cost_nodes(vehicle))
    random_order = generate_random_order(neighbourhoods)
    optimum_reached = False
    while optimum_reached == False:
        for i, neighbourhood in enumerate(random_order):
            customer_sequence_copy = best_nodes.copy()
            neighbourhood(customer_sequence_copy)
            calculated_nodes_cost = []
            for vehicle in customer_sequence_copy:
                if vehicle[0]!=vehicle[-1]:
                    vehicle[-1] = vehicle[0]
                calculated_nodes_cost.append(calculate_root_cost_nodes(vehicle))
            if max(best_nodes_cost)>max(calculated_nodes_cost):
                best_nodes = customer_sequence_copy
                best_nodes_cost = calculated_nodes_cost
                break
            else:
                if i == len(neighbourhoods) - 1:
                    optimum_reached = True

    return best_nodes, best_nodes_cost

def perturb(allocated_clusters, pert_rate):
    clusters_number = len(clusters)*pert_rate
    deleted_clusters = []
    i = 0
    while i < clusters_number:
        random_cluster = random.choice(clusters)
        for vehicle in allocated_clusters:
            for cluster in vehicle['clusters']:
                if cluster == random_cluster['ClusterID']:
                    deleted_clusters.append(random_cluster)
                    vehicle['clusters'].remove(cluster)
                    vehicle['remaining_capacity'] += random_cluster['TotalDemand']
        i+=1

    return allocated_clusters, deleted_clusters

def repair(allocated_clusters, deleted_clusters):
    for cluster in deleted_clusters:
        random_vehicle = random.randint(0, len(allocated_clusters) - 1)
        while allocated_clusters[random_vehicle]['remaining_capacity'] - cluster['TotalDemand'] < 0:
            random_vehicle = random.randint(0, len(allocated_clusters) - 1)
        random_spot = 0
        while random_spot == 0:
            random_spot = random.randint(0, len(allocated_clusters[random_vehicle]) - 1)
        allocated_clusters[random_vehicle]['clusters'].insert(random_spot,cluster['ClusterID'])
        allocated_clusters[random_vehicle]['remaining_capacity'] -= cluster['TotalDemand']
    
    return allocated_clusters

start = time.time()

# Stałe i zmienne
nIterationsNoImprovement = 0
goToNodeVNS = False
stoppingCriterion = False
maxIterationsNoImprovement = 1000
cluVNSProb = 0.5
best_clusters = None
best_cost = float('inf')
best_nodes_cost = float('inf')
vehicle_capacity = 550
counter = 0
best_clusters = []
best_nodes = []

# Step 0: Precomputation
xml_file_path = 'dataset.xml'

nodes = extract_data_from_xml(xml_file_path)
inter_cluster_distances = calculate_inter_cluster_distances(nodes)
cluster_nodes = create_nodes_clusters_table(nodes)
cluster_tsp_paths = {}  # Słownik przechowujący optymalne trasy dla każdego klastra

for cluster_id, cluster_customers in cluster_nodes.items():
    if cluster_id == 0:
        cluster_tsp_paths[cluster_id] = [0]
    else:
        tsp_path = solve_tsp_in_cluster(cluster_customers)
        cluster_tsp_paths[cluster_id] = tsp_path




# Step 1: Constructive phase
clusters = create_clusters_table(nodes)
allocated_clusters = allocate_clusters_to_vehicle(clusters, vehicle_capacity)

# Step 2: Intensification phase
while (counter==0) or (stoppingCriterion == False):
    counter += 1
    calculated_clusters, calculated_cost = vns_cluster_level(allocated_clusters, 100, inter_cluster_distances, best_clusters)
    first_run_node = True
    while (goToNodeVNS == True) or (first_run_node == True):
        first_run_node = False
        i = 1
        customer_sequence = convert_cluster_to_customer_sequence(nodes, cluster_tsp_paths, calculated_clusters, 0.4)
        calculated_nodes, calculated_nodes_cost = vns_customer_level(customer_sequence)
        for vehicle in calculated_nodes:
            if vehicle[0]!=vehicle[-1]:
                vehicle[-1] = vehicle[0]
        if max(calculated_nodes_cost)<best_nodes_cost:
                best_nodes_cost = max(calculated_nodes_cost)
                best_nodes = calculated_nodes
                nIterationsNoImprovement=0
        else:
            nIterationsNoImprovement += 1
            if nIterationsNoImprovement==maxIterationsNoImprovement:
                stoppingCriterion=True
                break
        
        #Step 3: Diversification phase
        allocated_clusters, deleted_clusters = perturb(allocated_clusters, 0.3)
        allocated_clusters = repair(allocated_clusters, deleted_clusters)
        r = random.randint(0,1)
        if r < cluVNSProb:
            goToNodeVNS = False
        else:
            goToNodeVNS = True



print(best_nodes)
print("Best cost:", best_nodes_cost)
customers = []
for vehicle in best_nodes:
    customers_vehicle = []
    for node in vehicle:
        customers_vehicle.append(node['id'])
    customers.append(customers_vehicle)

for vehicle in customers:
    print(vehicle)

nodes_allocated = []
for node in nodes:
    allocated = False
    for vehicle in customers:
        for node_allocated in vehicle:
            if node['id'] == node_allocated:
                allocated = True
                if node_allocated != 0:
                    nodes_allocated.append(node_allocated)
    if(allocated=="False"):
        print("Missing node:", node)
    
if len(nodes_allocated)>len(set(nodes_allocated)):
    print("Repeated values")

end = time.time()
print(end - start)
#execution time: 0.9477388858795166










        



