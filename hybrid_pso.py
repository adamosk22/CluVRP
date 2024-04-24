import copy
import math
import xml.etree.ElementTree as ET
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

def calculate_route_cost_clusters(clusters_data, inter_cluster_distances):
    # Funkcja obliczająca koszt trasy na podstawie danych klastrów
    total_cost = 0
    for i in range(len(clusters_data) - 1):
        cluster1 = clusters_data[i]
        cluster2 = clusters_data[i + 1]
        total_cost += inter_cluster_distances[cluster1][cluster2]
    return total_cost

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
    inter_cluster_distances = [[float('inf')] * n_clusters for _ in range(n_clusters)]
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            min_dist = min(euclidean_distance(p1, p2) for p1 in cluster_coords[i] for p2 in cluster_coords[j])
            inter_cluster_distances[i][j] = min_dist
            inter_cluster_distances[j][i] = min_dist
    
    return inter_cluster_distances

def calculate_route_cost_nodes(customers_data):
    total_cost = 0
    for i in range(len(customers_data) - 1):
        node1 = customers_data[i]
        node2 = customers_data[i+1]
        total_cost += euclidean_distance(node1, node2)
    return total_cost

# Function to calculate insertion cost
def calculate_insertion_cost(route, cluster, position):
    # Calculate increase in total route length
    route_with_cluster = route[:position] + [cluster] + route[position:]
    insertion_cost = calculate_route_cost_clusters(route_with_cluster, inter_cluster_distances) - calculate_route_cost_clusters(route, inter_cluster_distances)
    return insertion_cost

def calculate_insertion_cost_node(route, node, position):
    route_with_node = route[:position] + [node] + route[position:]
    insertion_cost = calculate_route_cost_nodes(route_with_node) - calculate_route_cost_nodes(route)
    return insertion_cost

# Function to insert cluster into route
def insert_cluster(route, cluster, position):
    return route[:position] + [cluster] + route[position:]

def insert_node(route, node, position):
    return route[:position] + [node] + route[position:]

def inter_shift1(allocated_nodes):
    if len(vehicles) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        cluster = random.choice(allocated_nodes[idx1])
        demand = clusters[cluster[0]['Cluster'] - 1]['TotalDemand']
        capacity = vehicles[idx2]['RemainingCapacity'] - demand
        if cluster[0]['Cluster'] != 0 and capacity > 0:
            idx = random.randint(1, len(allocated_nodes[idx2]) - 2)
            allocated_nodes[idx2].insert(idx, cluster)
            allocated_nodes[idx1].remove(cluster)
            vehicles[idx1]['RemainingCapacity'] += demand
            vehicles[idx2]['RemainingCapacity'] -= demand

def inter_shift2(allocated_nodes):
     if len(vehicles) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        i = random.randrange(1, len(allocated_nodes[idx1])-2)
        cluster1 = allocated_nodes[idx1][i]
        cluster2 = allocated_nodes[idx1][i+1]
        demand = clusters[cluster1[0]['Cluster'] - 1]['TotalDemand'] + clusters[cluster2[0]['Cluster'] - 1]['TotalDemand']
        capacity = vehicles[idx2]['RemainingCapacity'] - demand
        if cluster1[0]['Cluster'] != 0 and cluster2[0]['Cluster'] != 0 and capacity > 0:
            idx = random.randint(1, len(allocated_nodes[idx2]) - 2)
            allocated_nodes[idx2].insert(idx, cluster1)
            allocated_nodes[idx2].insert(idx + 1, cluster2)
            allocated_nodes[idx1].remove(cluster1)
            allocated_nodes[idx1].remove(cluster2)
            vehicles[idx1]['RemainingCapacity'] += demand
            vehicles[idx2]['RemainingCapacity'] -= demand


def inter_swap11(allocated_nodes):
# Operator Swap: zamienia pojazd dla dwóch klastrów
    if len(allocated_nodes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        i = random.randrange(1, len(allocated_nodes[idx1])-2)
        cluster1 = allocated_nodes[idx1][i]
        j = random.randrange(1, len(allocated_nodes[idx2])-2)
        cluster2 = allocated_nodes[idx2][j]
        demand1 = clusters[cluster1[0]['Cluster'] - 1]['TotalDemand']
        demand2 = clusters[cluster2[0]['Cluster'] - 1]['TotalDemand']
        capacity1 = vehicles[idx1]['RemainingCapacity'] + demand1 - demand2
        capacity2 = vehicles[idx2]['RemainingCapacity'] + demand2 - demand1
        if (capacity1 > 0) and (capacity2 > 0):
            allocated_nodes[idx1][i] = cluster2
            allocated_nodes[idx2][j] = cluster1
            vehicles[idx1]['RemainingCapacity'] = capacity1
            vehicles[idx2]['RemainingCapacity'] = capacity2

def inter_swap21(allocated_nodes):
# Operator Swap: zamienia pojazd dla dwóch klastrów
    if len(allocated_nodes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        i = random.randrange(1, len(allocated_nodes[idx1])-2)
        cluster11 = allocated_nodes[idx1][i]
        cluster12 = allocated_nodes[idx1][i+1]
        j = random.randrange(1, len(allocated_nodes[idx2])-2)
        cluster2 = allocated_nodes[idx2][j]
        demand1 = clusters[cluster11[0]['Cluster'] - 1]['TotalDemand'] + clusters[cluster12[0]['Cluster'] - 1]['TotalDemand']
        demand2 = clusters[cluster2[0]['Cluster'] - 1]['TotalDemand']
        capacity1 = vehicles[idx1]['RemainingCapacity'] + demand1 - demand2
        capacity2 = vehicles[idx1]['RemainingCapacity'] + demand2 - demand1
        if (capacity1 > 0) and (capacity2 > 0):
            allocated_nodes[idx1][i] = cluster2
            allocated_nodes[idx2][j] = cluster11
            allocated_nodes[idx1].remove(cluster12)
            allocated_nodes[idx2].insert(j+1, cluster12)
            vehicles[idx1]['RemainingCapacity'] = capacity1
            vehicles[idx2]['RemainingCapacity'] = capacity2

def inter_swap22(allocated_nodes):
# Operator Swap: zamienia pojazd dla dwóch klastrów
    if len(allocated_nodes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        i = random.randrange(1, len(allocated_nodes[idx1])-2)
        cluster11 = allocated_nodes[idx1][i]
        cluster12 = allocated_nodes[idx1][i+1]
        j = random.randrange(1, len(allocated_nodes[idx2])-2)
        cluster21 = allocated_nodes[idx2][j]
        cluster22 = allocated_nodes[idx2][j+1]
        demand1 = clusters[cluster11[0]['Cluster'] - 1]['TotalDemand'] + clusters[cluster12[0]['Cluster'] - 1]['TotalDemand']
        demand2 = clusters[cluster21[0]['Cluster'] - 1]['TotalDemand'] + clusters[cluster22[0]['Cluster'] - 1]['TotalDemand']
        capacity1 = vehicles[idx1]['RemainingCapacity'] + demand1 - demand2
        capacity2 = vehicles[idx1]['RemainingCapacity'] + demand2 - demand1
        if (capacity1 > 0) and (capacity2 > 0):
            allocated_nodes[idx1][i] = cluster21
            allocated_nodes[idx2][j] = cluster11
            allocated_nodes[idx1][i+1] = cluster22
            allocated_nodes[idx2][j+1] = cluster12
            vehicles[idx1]['RemainingCapacity'] = capacity1
            vehicles[idx2]['RemainingCapacity'] = capacity2

def inter_2_opt(allocated_nodes):
    if len(allocated_nodes) >= 2:
        idx1, idx2 = random.sample(range(len(vehicles)), 2)
        i = random.randrange(1, len(allocated_nodes[idx1])-2)
        clusters1 = allocated_nodes[idx1][i:]
        j = random.randrange(1, len(allocated_nodes[idx2])-2)
        clusters2 = allocated_nodes[idx2][j:]
        demand1 = 0
        for cluster in clusters1:
            demand1 += clusters[cluster[0]['Cluster'] - 1]['TotalDemand']
        demand2 = 0
        for cluster in clusters2:
            demand2 += clusters[cluster[0]['Cluster'] - 1]['TotalDemand']
        capacity1 = vehicles[idx2]['RemainingCapacity'] + demand1 - demand2
        capacity2 = vehicles[idx2]['RemainingCapacity'] + demand2 - demand1
        if capacity1 > 0 and capacity2 > 0:
            for cluster in clusters1:
                allocated_nodes[idx1].remove(cluster)
            for cluster in clusters2:
                allocated_nodes[idx2].remove(cluster)
            for cluster in clusters1:
                allocated_nodes[idx2].append(cluster)
            for cluster in clusters2:
                allocated_nodes[idx1].append(cluster)
            for cluster in allocated_nodes[idx1]:
                if cluster[0]['id'] == 0:
                    allocated_nodes[idx1].remove(cluster)
            for cluster in allocated_nodes[idx2]:
                if cluster[0]['id'] == 0:
                    allocated_nodes[idx2].remove(cluster)
            for vehicle in allocated_nodes:
                vehicle.insert(0, [{'Addr': '0', 'id': 0, 'Cluster': 0, 'DemDelivery': 0, 'DemPickup': 0, 'CoordX': 499090, 'CoordY': 4792870}])
                vehicle.append([{'Addr': '0', 'id': 0, 'Cluster': 0, 'DemDelivery': 0, 'DemPickup': 0, 'CoordX': 499090, 'CoordY': 4792870}])
            vehicles[idx1]['RemainingCapacity'] = capacity1
            vehicles[idx2]['RemainingCapacity'] = capacity2


def intra_shift1(allocated_nodes, idx1):
    cluster = random.choice(allocated_nodes[idx1])
    if cluster[0]['Cluster'] != 0:
        allocated_nodes[idx1].remove(cluster)
        if len(allocated_nodes[idx1]) > 4:
            idx = random.randint(1, len(allocated_nodes[idx1]) - 2)
        else: 
            idx = 1
        allocated_nodes[idx1].insert(idx, cluster)

def intra_swap(allocated_nodes, idx1):
    i = random.randrange(len(allocated_nodes[idx1])-2)
    cluster1 = allocated_nodes[idx1][i]
    j = random.randrange(len(allocated_nodes[idx1])-2)
    cluster2 = allocated_nodes[idx1][j]
    if cluster1[0]['Cluster'] != 0 and cluster2[0]['Cluster'] != 0:
        allocated_nodes[idx1][i] = cluster2
        allocated_nodes[idx1][j] = cluster1

def intra_or_opt2(allocated_nodes, idx1):
    if len(allocated_nodes[idx1]) > 4:
        i = random.randrange(len(allocated_nodes[idx1])-2)
        cluster1 = allocated_nodes[idx1][i]
        cluster2 = allocated_nodes[idx1][i+1]
        if cluster1[0]['Cluster'] != 0 and cluster2[0]['Cluster'] != 0:
            allocated_nodes[idx1].remove(cluster1)
            allocated_nodes[idx1].remove(cluster2)
            if len(allocated_nodes[idx1]) > 4:
                idx = random.randint(1, len(allocated_nodes[idx1]) - 2)
            else: 
                idx = 1
            allocated_nodes[idx1].insert(idx, cluster1)
            allocated_nodes[idx1].insert(idx + 1, cluster2)

def intra_or_opt3(allocated_nodes, idx1):
    i = random.randrange(len(allocated_nodes[idx1])-2)
    cluster1 = allocated_nodes[idx1][i]
    cluster2 = allocated_nodes[idx1][i+1]
    cluster3 = allocated_nodes[idx1][i+2]
    if cluster1[0]['Cluster'] != 0 and cluster2[0]['Cluster'] != 0 and cluster3[0]['Cluster'] != 0:
        allocated_nodes[idx1].remove(cluster1)
        allocated_nodes[idx1].remove(cluster2)
        allocated_nodes[idx1].remove(cluster3)
        if len(allocated_nodes[idx1]) > 4:
            idx = random.randint(1, len(allocated_nodes[idx1]) - 2)
        else: 
            idx = 1
        allocated_nodes[idx1].insert(idx, cluster1)
        allocated_nodes[idx1].insert(idx + 1, cluster2)
        allocated_nodes[idx1].insert(idx + 2, cluster2)

def intra_2_opt(allocated_nodes, idx1):
    i = random.randrange(len(allocated_nodes[idx1])-2)
    cluster11 = allocated_nodes[idx1][i]
    cluster12 = allocated_nodes[idx1][i+1]
    j = random.randrange(len(allocated_nodes[idx1])-2)
    cluster21 = allocated_nodes[idx1][j]
    cluster22 = allocated_nodes[idx1][j+1]
    if cluster11[0]['Cluster'] != 0 and cluster12[0]['Cluster'] != 0 and cluster21[0]['Cluster'] != 0 and cluster22[0]['Cluster'] != 0:
        allocated_nodes[idx1].remove(cluster11)
        allocated_nodes[idx1].remove(cluster12)
        if cluster11 != cluster21 and cluster12 != cluster21:
            allocated_nodes[idx1].remove(cluster21)
            allocated_nodes[idx1].insert(i+1, cluster21)
        if cluster11 != cluster22 and cluster12 != cluster22:
            allocated_nodes[idx1].remove(cluster22)
            allocated_nodes[idx1].insert(i, cluster22)
        allocated_nodes[idx1].insert(j, cluster12)
        allocated_nodes[idx1].insert(j+1, cluster11)

def intra_shift_nodes(allocated_nodes, idx1, cluster_number):
    cluster = allocated_nodes[idx1][cluster_number]
    if cluster[0]['id'] != 0:
        i = random.randrange(len(cluster)-2)
        node = cluster[i]
        if node['id'] != 0:
            cluster.remove(node)
            idx = random.randint(1, len(cluster) - 2)
            cluster.insert(idx, node)
            allocated_nodes[idx1][cluster_number] = cluster

def intra_swap_nodes(allocated_nodes, idx1, cluster_number):
    cluster = allocated_nodes[idx1][cluster_number]
    if cluster[0]['id'] != 0:
        i = random.randrange(len(cluster)-2)
        node1 = cluster[i]
        j = random.randrange(len(cluster)-2)
        node2 = cluster[j]
        if node1['id'] != 0 and node2['id'] != 0:
            cluster[j] = node1
            cluster[i] = node2
            allocated_nodes[idx1][cluster_number] = cluster

def intra_2opt_nodes(allocated_nodes, idx1, cluster_number):
    cluster = allocated_nodes[idx1][cluster_number]
    if cluster[0]['id'] != 0:
        i = random.randrange(len(cluster)-2)
        node11 = cluster[i]
        node12 = cluster[i+1]
        j = 0
        while j == 0 or j == i or j == i+1 or j == i-1:
            j = random.randrange(len(cluster)-2)
        node21 = cluster[j]
        node22 = cluster[j+1]
        if node11['id'] != 0 and node21['id'] != 0 and node12['id'] != 0 and node22['id'] != 0:
            cluster.remove(node11)
            cluster.remove(node12)
            cluster.remove(node21)
            cluster.remove(node22)
            cluster.insert(i, node22)
            cluster.insert(i+1, node21)
            cluster.insert(j, node12)
            cluster.insert(j+1, node11)
            allocated_nodes[idx1][cluster_number] = cluster

def generate_clusters_route(clusters, position_values, vehicles):
    cluster_order = sorted(range(len(position_values)), key=lambda i: position_values[i], reverse=True)
    cluster_order.remove(0)
    i = 0
    for vehicle in vehicles:
        vehicle['Clusters'].append(0)
        vehicle['Clusters'].append(0)
    for cluster in cluster_order:
        min_insertion_cost_vehicle = []
        min_insertion_position_vehicle = []
        i = 0
        for vehicle in vehicles:
            # Evaluate insertion cost for vehicle
            min_insertion_cost_vehicle.append(float('inf'))
            min_insertion_position_vehicle.append(None)
            initial_route_vehicle = vehicle['Clusters'].copy()
            for position in range(len(initial_route_vehicle) + 1):
                if position != 0 and position != len(initial_route_vehicle):
                    insertion_cost = calculate_insertion_cost(initial_route_vehicle, cluster, position)
                    if insertion_cost < min_insertion_cost_vehicle[i]:
                        min_insertion_cost_vehicle[i] = insertion_cost
                        min_insertion_position_vehicle[i] = position
            i += 1


        vehicle_index = min_insertion_cost_vehicle.index(min(min_insertion_cost_vehicle))
        while (vehicles[vehicle_index]['RemainingCapacity'] - clusters[cluster - 1]['TotalDemand']) < 0:
            min_insertion_cost_vehicle[vehicle_index] = float('inf')
            vehicle_index = min_insertion_cost_vehicle.index(min(min_insertion_cost_vehicle))
        vehicles[vehicle_index]['Clusters'] = insert_cluster(vehicles[vehicle_index]['Clusters'], cluster, min_insertion_position_vehicle[vehicle_index])
        vehicles[vehicle_index]['RemainingCapacity'] -= clusters[cluster - 1]['TotalDemand']
    
    return vehicles


def generate_customers_route(vehicles, nodes, position_values):
    node_order = sorted(range(len(position_values)), key=lambda i: position_values[i], reverse=True)
    for vehicle in vehicles:
        vehicle['Path'].append([{'Addr': '0', 'id': 0, 'Cluster': 0, 'DemDelivery': 0, 'DemPickup': 0, 'CoordX': 499090, 'CoordY': 4792870}])
        for cluster in vehicle['Clusters']:
            if cluster != 0:
                vehicle['Path'].append([])
        vehicle['Path'].append([{'Addr': '0', 'id': 0, 'Cluster': 0, 'DemDelivery': 0, 'DemPickup': 0, 'CoordX': 499090, 'CoordY': 4792870}])
    for node_index in node_order:
        node = None
        for n in nodes:
            if n['id'] == node_index:
                node = n
        for vehicle in vehicles:
            i = 0
            for cluster in vehicle['Clusters']:
                if cluster == node['Cluster'] and cluster != 0:
                    min_insertion_cost = float('inf')
                    min_insertion_position = None
                    initial_route_vehicle = vehicle['Path'][i].copy()
                    for position in range(len(initial_route_vehicle) + 1):
                        insertion_cost = calculate_insertion_cost_node(initial_route_vehicle, node, position)
                        if insertion_cost < min_insertion_cost:
                            min_insertion_cost = insertion_cost
                            min_insertion_position = position
                    vehicle['Path'][i] = insert_node(vehicle['Path'][i], node, min_insertion_position)
                i += 1
    for vehicle in vehicles:
            distance = 0
            i = 0
            for cluster in vehicle['Path']:
                j = 0
                while j < len(cluster) - 1:
                    distance += euclidean_distance(cluster[j], cluster[j+1])
                    j += 1
                if i < len(vehicle['Path']) - 1:
                    distance += euclidean_distance(vehicle['Path'][i][-1], vehicle['Path'][i+1][0])
                i += 1
    return vehicles

def calculate_distance(vehicles):
    distances = []
    for vehicle in vehicles:
        distances.append(calculate_distance_vehicle(vehicle))
    return distances

def calculate_distance_vehicle(vehicle):
    distance = 0
    i = 0
    for cluster in vehicle:
        distance += calculate_distance_cluster(cluster)
        if i < len(vehicle) - 1:
            distance += euclidean_distance(vehicle[i][-1], vehicle[i+1][0])
        i += 1
    return distance

def calculate_distance_cluster(cluster):
    j = 0
    distance = 0
    while j < len(cluster) - 1:
        distance += euclidean_distance(cluster[j], cluster[j+1])
        j += 1
    return distance

def vns(s):
    s_initial = copy.deepcopy(s)
    first_iterartion = True
    i = 0
    while max(calculate_distance(s)) < max(calculate_distance(s_initial)) or first_iterartion == True:
        i+=1
        first_iterartion = False
        s_initial = copy.deepcopy(s)
        NLc = [inter_shift1, inter_shift2, inter_swap11, inter_swap21, inter_swap22]
        NLc_full = [inter_shift1, inter_shift2, inter_swap11, inter_swap21, inter_swap22]
        s_best = []
        distances_best = [float('inf'), float('inf')]
        while len(NLc) != 0:
            neighbourhood = random.choice(NLc)
            s_copy = copy.deepcopy(s)
            for vehicle in s_copy:
                for cluster in vehicle:
                    print("vehicle", vehicle)
            neighbourhood(s_copy)
            for vehicle in s_copy:
                clusters_vehicle = []
                for cluster in vehicle:
                    clusters_vehicle.append(cluster[0]['Cluster'])
                if len(clusters_vehicle)!=(len(set(clusters_vehicle)) + 1) or clusters_vehicle[0]!=0 or clusters_vehicle[-1]!=0:
                    s_copy = copy.deepcopy(s)
            len_s = 0
            len_s_copy = 0
            for vehicle in s:
                len_s += len(vehicle)
            for vehicle in s_copy:
                len_s_copy += len(vehicle)
            if(len_s != len_s_copy):
                print("pryczyna", neighbourhood.__name__)
            distances_after = calculate_distance(s_copy)
            if max(distances_after) < max(distances_best):
                print("ulepszono")
                s = copy.deepcopy(s_copy)
                s = intra_route_search(s)
                s_best = copy.deepcopy(s_copy)
                distances_best = distances_after
                NLc = NLc_full.copy()
            else:
                NLc.remove(neighbourhood)
                print(distances_after)
                print(NLc)
        print(s)
        s = intra_cluster_search(s)
    return s_best

def intra_route_search(s_vehicles):
    i = 0
    for s in s_vehicles:
        s_initial = copy.deepcopy(s)
        first_iterartion = True
        distance_best = float('inf')
        s_best_vehicle = []
        while calculate_distance_vehicle(s) < calculate_distance_vehicle(s_initial) or first_iterartion == True:
            first_iterartion = False
            s_initial = copy.deepcopy(s)
            NLc = [intra_or_opt2, intra_or_opt3, intra_shift1, intra_swap, intra_2_opt]
            NLc_full = [intra_or_opt2, intra_or_opt3, intra_shift1, intra_swap, intra_2_opt]
            while len(NLc) != 0:
                neighbourhood = random.choice(NLc)
                s_copy = copy.deepcopy(s_vehicles)
                l1 = len(s_vehicles[i])
                neighbourhood(s_copy, i)
                for vehicle in s_copy:
                    clusters_vehicle = []
                    for cluster in vehicle:
                        clusters_vehicle.append(cluster[0]['Cluster'])
                    if len(clusters_vehicle)!=(len(set(clusters_vehicle)) + 1) or clusters_vehicle[0]!=0 or clusters_vehicle[-1]!=0:
                        s_copy = copy.deepcopy(s_vehicles)
                l2 = len(s_copy[i])
                if l1 != l2:
                    print('przyczyna', neighbourhood.__name__)
                distance_after = calculate_distance_vehicle(s_copy[i])
                if distance_after < distance_best:
                    s_vehicles = copy.deepcopy(s_copy)
                    s = copy.deepcopy(s_copy[i])
                    NLc = NLc_full
                    s_best_vehicle = copy.deepcopy(s_copy[i])
                    distance_best = distance_after
                else:
                    NLc.remove(neighbourhood)
        s_vehicles[i] = s_best_vehicle
        i += 1
    return s_vehicles

def intra_cluster_search(s_vehicles):
    i = 0
    for s_vehicle in s_vehicles:
        j = 0
        for s in s_vehicle:
            distance_best = float('inf')
            s_best_cluster = []
            s_initial = copy.deepcopy(s)
            first_iterartion = True
            while calculate_distance_cluster(s) < calculate_distance_cluster(s_best_cluster) or first_iterartion == True:
                first_iterartion = False
                NLc = [intra_2opt_nodes, intra_shift_nodes, intra_swap_nodes]
                NLc_full = [intra_2opt_nodes, intra_shift_nodes, intra_swap_nodes]
                while len(NLc) != 0:
                    neighbourhood = random.choice(NLc)
                    s_copy = [x[:] for x in s_vehicles]
                    neighbourhood(s_copy, i, j)
                    distance_after = calculate_distance_cluster(s_copy[i][j])
                    if distance_after < distance_best:
                        s_vehicles = [x[:] for x in s_copy]
                        s = copy.copy(s_copy[i][j])
                        s_best_cluster = copy.copy(s_copy[i][j])
                        NLc = NLc_full
                        distance_best = distance_after
                    else:
                        NLc.remove(neighbourhood)
            s_vehicles[i][j] = s_best_cluster
            j += 1
        i += 1
    return s_vehicles






w = 0.7
c1 = c2 = 2
r1 = r2 = 0.5
alfa_max = beta_max = gamma_max = delta_max = 4
alfa_min = beta_min = gamma_min = delta_min = -4
vehicle_capacity = 600

xml_file_path = 'dataset.xml'
nodes = extract_data_from_xml(xml_file_path)
inter_cluster_distances = calculate_inter_cluster_distances(nodes)
n = len(nodes)
c = 0
current_cluster = -1
for node in nodes:
    if current_cluster != node['Cluster']:
        current_cluster = node['Cluster']
        c += 1

K = n/4
X = []
Y = []
U = []
V = []
fb = []
i = 0
while i < K:
    X_row = []
    Y_row = [] 
    U_row = [] 
    V_row = []
    j = 0
    while j < n:
        gamma_i_l = gamma_min + (gamma_max - gamma_min)*random.uniform(0,1)
        X_row.append(gamma_i_l)
        beta_i_l = beta_min + (beta_max - beta_min)*random.uniform(0,1)
        U_row.append(beta_i_l)
        j+=1
    j = 0
    while j < c:
        alfa_i_l = alfa_min + (alfa_max - alfa_min)*random.uniform(0,1)
        Y_row.append(alfa_i_l)
        delta_i_l = delta_min + (delta_max - delta_min)*random.uniform(0,1)
        V_row.append(delta_i_l)
        j+=1
    X.append(X_row)
    Y.append(Y_row)
    U.append(U_row)
    V.append(V_row)
    fb.append(float('inf'))
    i+=1
fg = float('inf')
#main phase
clusters = create_clusters_table(nodes)
full_demand = 0
for cluster in clusters:
    full_demand += cluster['TotalDemand']
number_of_vehicles = full_demand/vehicle_capacity
print(number_of_vehicles)
s_best = []
distances_best = [float('inf'), float('inf')]
i = 0
while i < len(Y):
    vehicles = []
    j = 0
    while j < number_of_vehicles:
        vehicles.append({"Clusters": [], "RemainingCapacity": vehicle_capacity, "Path": []})
        j += 1
    print("ite", i)
    vehicles = generate_clusters_route(clusters, Y[i], vehicles)
    vehicles = generate_customers_route(vehicles, nodes, X[i])
    allocated_nodes = []
    for vehicle in vehicles:
        allocated_nodes.append(vehicle['Path'])
    distance_before = calculate_distance(allocated_nodes)
    print("before", distance_before)
    s = vns(allocated_nodes)
    print("after", calculate_distance(s))
    if max(calculate_distance(s)) < max(distances_best):
        s_best = copy.deepcopy(s)
        distances_best = calculate_distance(s)
        print("new", distances_best)
    i += 1
for vehicle in s_best:
    for cluster in vehicle:
        print(cluster[0]['Cluster'])
print("nodes", s_best)
print(distances_best)



    







