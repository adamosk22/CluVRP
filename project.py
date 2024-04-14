import xml.etree.ElementTree as ET
import numpy as np
import sys
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

def tsp_nearest_neighbor(coordinates, number):
    num_nodes = len(coordinates)
    visited = [False] * num_nodes

    start_node = 0
    visited[start_node] = True
    current_node = start_node
    path = [current_node]
    total_distance = 0


    for _ in range(num_nodes - 1):
        nearest_neighbor = None
        nearest_distance = sys.float_info.max

        for neighbor_node in range(num_nodes):
            if not visited[neighbor_node]:
                distance = calculate_distance(coordinates[current_node], coordinates[neighbor_node])

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_neighbor = neighbor_node

        path.append(nearest_neighbor)
        total_distance += nearest_distance
        visited[nearest_neighbor] = True
        current_node = nearest_neighbor


    path_global = []

    for node in path:
        format_node = str(number) + "." + str(node + 1)
        path_global.append(format_node)
        
    return path_global, total_distance, number, coordinates[0], coordinates[path[-1]]

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
    
def tsp_nearest_neighbor_global(coordinates, vehicles, clusters):
    num_nodes = len(coordinates)
    num_nodes_left = num_nodes
    visited = [False] * num_nodes

    start_node = 0
    visited[start_node] = True
    current_node = [start_node, start_node]
    for vehicle in vehicles:
        vehicle['Clusters'].append(current_node[0])
        vehicle['Path'] = [current_node[0]]
    total_distance = [0,0]
    vehicle_number = 0


    while num_nodes_left > 1:
        nearest_neighbor = None
        nearest_distance = sys.float_info.max

        for neighbor_node in range(num_nodes):
            if not visited[neighbor_node]:
                distance = calculate_distance(coordinates[current_node[vehicle_number]][0], coordinates[neighbor_node][0])
                if coordinates[current_node[vehicle_number]][1] == coordinates[neighbor_node][1]:
                    distance = 0

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_neighbor = neighbor_node

        if(coordinates[nearest_neighbor][1] not in vehicle['Path']):
            vehicle['Path'].append(coordinates[nearest_neighbor][1])
            vehicles[vehicle_number]['Clusters'].append(coordinates[nearest_neighbor][1])
            for cluster in clusters:
                if cluster['ClusterID'] == coordinates[nearest_neighbor][1]:
                    vehicles[vehicle_number]['RemainingCapacity'] -= cluster['TotalDemand']
            if vehicle_number == (len(vehicles) - 1):
                vehicle_number = 0
            else:
                vehicle_number += 1
        total_distance[vehicle_number] += nearest_distance
        visited[nearest_neighbor] = True
        num_nodes_left -= 1
        if num_nodes_left == 1:
            break
        current_node[vehicle_number] = nearest_neighbor
    vehicle_number = 0
    for vehicle in vehicles:
        last_node = vehicle['Path'][-1]
        distance_to_start = calculate_distance(coordinates[last_node][0], coordinates[start_node][0])
        total_distance[vehicle_number] += distance_to_start
        vehicle['Clusters'].append(start_node)
        vehicle['Path'].append(start_node)
        vehicle_number += 1

    return vehicles, total_distance

def find_path(extracted_data):
    paths = []
    vehicles = []
    clusters = create_clusters_table(extracted_data)
    full_demand = 0
    for cluster in clusters:
        full_demand += cluster['TotalDemand']
    number_of_vehicles = full_demand/vehicle_capacity
    i = 0
    while i < number_of_vehicles:
        vehicles.append({"Clusters": [], "RemainingCapacity": vehicle_capacity, "Path": []})
        i += 1
    cluster_number = -1
    for node in extracted_data:
        if(node['Cluster'] > cluster_number):
            cluster_number = node['Cluster']
            cluster_coordinates = find_coordinates(extracted_data, cluster_number)
            if node['Cluster'] != 0:
                cluster_path = tsp_nearest_neighbor(cluster_coordinates, cluster_number)
                if(len(cluster_path[0]) != 10):
                    print("Nodes missing")
                    print(cluster_path)
                paths.append(cluster_path)
            else:
                paths.append([[0], 0, 0, cluster_coordinates[0], cluster_coordinates[0]])
    base_coordinates = []
    distance = 0
    for path in paths:
        base_coordinates.append([path[3], path[2]])
        base_coordinates.append([path[4], path[2]])
    base_result, base_distance = tsp_nearest_neighbor_global(base_coordinates, vehicles, clusters)
    i = 0
    for vehicle in base_result:
        base_path = vehicle['Clusters']
        distance = base_distance[i]
        for path in paths:
            if path[2] in vehicle['Clusters']:
                distance += path[1]
        print("Distance:", distance)
        vehicle['Path'] = []
        for node in base_path:
            vehicle['Path'].append(paths[node][0])
        print(vehicle)
        i+=1
        
def find_coordinates(extracted_data, number):
    coordinates = []
    for node in extracted_data:
        if node['Cluster'] == number:
            coord = (node['CoordX'], node['CoordY'])
            coordinates.append(coord)
    return coordinates

def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

start = time.time()

xml_file_path = 'dataset.xml'

extracted_data = extract_data_from_xml(xml_file_path)

vehicle_capacity = 500

find_path(extracted_data)

end = time.time()
print('exexution time', end - start)


#exexution time 0.003947257995605469





