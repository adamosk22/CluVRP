import copy
import itertools
import math
import os
import statistics
import sys
import time
import xml.etree.ElementTree as ET
import random
import networkx as nx

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
    i = 0
    for node in data:
        cluster_coords[node['Cluster']].append({'CoordX': node['CoordX'], 'CoordY': node['CoordY']})

    # Obliczanie najkrótszych odległości między klastrami
    inter_cluster_distances = [[math.inf] * n_clusters for _ in range(n_clusters)]
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if cluster_coords[i] != [] and cluster_coords[j] != []:
                min_dist = min(euclidean_distance(p1, p2) for p1 in cluster_coords[i] for p2 in cluster_coords[j])
                inter_cluster_distances[i][j] = min_dist
                inter_cluster_distances[j][i] = min_dist
            else:
                inter_cluster_distances[i][j] = 0
                inter_cluster_distances[j][i] = 0
    
    return inter_cluster_distances

def generate_random_routes(clusters, population_size):
    routes = []
    i = 0
    while i < population_size:
        route = []
        for cluster in clusters:
            if cluster['ClusterID'] != 0:
                route.append(cluster)
        random.shuffle(route)
        j = 0
        remaining_capacity = vehicle_capacity
        while j < len(route) - 1:
            remaining_capacity -= route[j]['TotalDemand']
            if remaining_capacity < 0:
                route.insert(j-1, {'ClusterID': 0, 'TotalDemand': 0})
                remaining_capacity = vehicle_capacity

            j += 1
        routes.append(route)
        i += 1
    return routes

def calculate_route_cost_clusters(clusters_data):
    # Funkcja obliczająca koszt trasy na podstawie danych klastrów
    costs = []
    total_cost = 0
    for i in range(len(clusters_data) - 1):
        cluster1 = clusters_data[i]['ClusterID']
        cluster2 = clusters_data[i + 1]['ClusterID']
        total_cost += inter_cluster_distances[cluster1][cluster2]
        if cluster2 == 0:
            costs.append(total_cost)
            total_cost = 0
    costs.append(total_cost)
    if inter_cluster_distances[0][clusters_data[0]['ClusterID']] != math.inf:
        costs[0] += inter_cluster_distances[0][clusters_data[0]['ClusterID']]
    if inter_cluster_distances[0][clusters_data[-1]['ClusterID']] != math.inf:
        costs[-1] += inter_cluster_distances[0][clusters_data[-1]['ClusterID']]
    return max(costs)

def crossover(parent1, parent2):
    # Funkcja krzyżująca dwie trasy
    cross1 = random.randrange(0, len(parent1) - 1)
    cross2 = random.randrange(cross1, len(parent1) - 1)
    child1 = [{'ClusterID': -1, 'TotalDemand': 0}] * len(parent1)
    child2 = [{'ClusterID': -1, 'TotalDemand': 0}] * len(parent1)
    n_zeroes = 0
    for cluster in parent1:
        if cluster['ClusterID'] == 0:
            n_zeroes += 1
    i = cross1
    while i < cross2:
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        i+=1
    n_zeroes1 = 0
    for cluster in child1:
        if cluster['ClusterID'] == 0:
            n_zeroes1 += 1
    n_zeroes2 = 0
    for cluster in child2:
        if cluster['ClusterID'] == 0:
            n_zeroes2 += 1
    i = 0
    for cluster in parent2:
        if cluster not in child1 or (cluster['ClusterID'] == 0 and n_zeroes != n_zeroes1):
            while child1[i]['ClusterID'] != -1 and i < len(child1) -1:
                i += 1
            child1[i] = cluster
            if cluster['ClusterID'] == 0:
                n_zeroes1 += 1
            i += 1
    i = 0
    for cluster in parent1:
        if cluster not in child2 or (cluster == 0 and n_zeroes != n_zeroes2):
            while child2[i]['ClusterID'] != -1 and i < len(child2) -1:
                i += 1
            child2[i] = cluster
            if cluster['ClusterID'] == 0:
                n_zeroes2 += 1
            i += 1
    rand = random.randrange(0,1)
    if rand < mutation_prob:
        cluster1 = random.choice(child1)
        cluster2 = random.choice(child1)
        for cluster in child1:
            if cluster == cluster1:
                cluster = cluster2
            if cluster == cluster2:
                cluster = cluster1
        cluster1 = random.choice(child2)
        cluster2 = random.choice(child2)
        for cluster in child2:
            if cluster == cluster1:
                cluster = cluster2
            if cluster == cluster2:
                cluster = cluster1
    return child1, child2

def genetic_algorithm(routes, n_epochs):
    k = 0
    while k != n_epochs:
        # Sortowanie tras według kosztu
        sorted_routes = sorted(routes, key=calculate_route_cost_clusters)

        # Zachowanie najlepszej trasy
        best_route = sorted_routes[0]
        new_routes = [best_route]

        # Selekcja rodziców za pomocą ruletkowego wyboru
        total_cost = sum(calculate_route_cost_clusters(route) for route in sorted_routes)
        accumulated_ranks = [sum(calculate_route_cost_clusters(route) for route in sorted_routes[:i+1]) for i in range(len(sorted_routes))]
        while total_cost == math.inf:
            del accumulated_ranks[-1]
            del sorted_routes[-1]
            total_cost = sum(calculate_route_cost_clusters(route) for route in sorted_routes)
            accumulated_ranks = [sum(calculate_route_cost_clusters(route) for route in sorted_routes[:i+1]) for i in range(len(sorted_routes))]
            print(accumulated_ranks[-1])

        while len(new_routes) < len(routes):
            parent1 = select_parent(sorted_routes, accumulated_ranks, total_cost)
            parent2 = select_parent(sorted_routes, accumulated_ranks, total_cost)
            if parent1 != parent2:
                child1, child2 = crossover(parent1, parent2)
                j = 0
                remaining_capacity = vehicle_capacity
                while j < len(child1) - 1:
                    remaining_capacity -= child1[j]['TotalDemand']
                    if child1[j]['ClusterID'] == 0:
                        remaining_capacity = vehicle_capacity
                    j += 1
                if remaining_capacity >= 0:
                    new_routes.append(child1)
                remaining_capacity = vehicle_capacity
                j = 0
                while j < len(child2) - 1:
                    remaining_capacity -= child2[j]['TotalDemand']
                    if child2[j]['ClusterID'] == 0:
                        remaining_capacity = vehicle_capacity
                    j += 1
                if remaining_capacity >= 0:
                    new_routes.append(child2)
        routes = new_routes[:len(routes)]
        k += 1

    i = 0
    first_run = True
    capacity_exceeded = False
    while capacity_exceeded or first_run:
        first_run = False
        capacity_exceeded = False
        chosen_route = sorted(routes, key=calculate_route_cost_clusters)[i]
        demand = 0
        for cluster in chosen_route:
            demand += cluster['TotalDemand']
            if demand > vehicle_capacity:
                capacity_exceeded = True
                i += 1
                print(demand)
                print(chosen_route)
            if cluster['ClusterID'] == 0:
                demand = 0
        print(capacity_exceeded)
    return chosen_route

def select_parent(routes, accumulated_ranks, total_cost):
    R = random.uniform(0, total_cost)
    for i, rank in enumerate(accumulated_ranks):
        if rank > R:
            return routes[i]

def generate_customer_routes(cluster_route, nodes):
    routes = []
    customers_groups = []
    current_group = []
    
    # Grupowanie klientów na podstawie klastrów
    for cluster in cluster_route:
        for node in nodes:
            if node['Cluster'] == cluster['ClusterID']:
                current_group.append(node)
        if cluster['ClusterID'] == 0 and current_group:
            customers_groups.append(copy.deepcopy(current_group))
            current_group = []
    if current_group:
        customers_groups.append(copy.deepcopy(current_group))

    for group in customers_groups:
        if nodes[0] not in group:
            group.append(nodes[0])

        # Tworzenie grafu dla klientów w klastrze
        G = nx.Graph()
        for customer in group:
            G.add_node(customer['id'], pos=(customer['CoordX'], customer['CoordY']))

        # Dodanie krawędzi między klientami w grafie na podstawie odległości (np. odległość euklidesowa)
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                dist = ((group[i]['CoordX'] - group[j]['CoordX']) ** 2 + 
                        (group[i]['CoordY'] - group[j]['CoordY']) ** 2) ** 0.5
                if group[i]['Cluster'] != group[j]['Cluster']:
                    dist += 1e10  # Dodanie dużej kary dla różnych klastrów
                G.add_edge(group[i]['id'], group[j]['id'], weight=dist)

        # Wyznaczanie ścieżki komiwojażera
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method=nx.approximation.greedy_tsp) 
        zero_index = tsp_path.index(0)
        route = tsp_path[zero_index:] + tsp_path[:zero_index]
        route.append(0)
        routes.append(route)
        
    print(routes)
    return routes
    
        
def calculate_route_cost_customers(customer_routes, nodes):
    distances = []
    for route in customer_routes:
        distance = 0
        for i in range(len(route) - 1):
            distance += euclidean_distance(nodes[route[i]], nodes[route[i+1]])
    
        distances.append(distance)
    return distances







start = time.time()
vehicle_capacity = 600
n_epochs = 200
population_size = 70
mutation_prob = 0.2
times = []
costs = []
if __name__ == "__main__":
    n_epochs = int(sys.argv[1])
    population_size = int(sys.argv[2])
    mutation_prob = float(sys.argv[3])

folder_path = 'instances'
files = os.listdir(folder_path)
for file_name in files:
    if file_name.endswith('.xml'):
        start = time.time()
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        nodes = extract_data_from_xml(file_path)
        inter_cluster_distances = calculate_inter_cluster_distances(nodes)
        clusters = create_clusters_table(nodes)
        routes = generate_random_routes(clusters, population_size)
        cluster_route = genetic_algorithm(routes,n_epochs)
        print(cluster_route)
        customer_routes = generate_customer_routes(cluster_route, nodes)
        distances = calculate_route_cost_customers(customer_routes, nodes)
        print(distances)
        end = time.time()
        print(end - start)
        times.append(end-start)
        costs.append(max(distances))
print(times)
print(costs)
print(round(statistics.mean(times),2))
print(round(statistics.mean(costs),2))
        


