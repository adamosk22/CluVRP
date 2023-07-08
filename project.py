import xml.etree.ElementTree as ET
import numpy as np
import sys



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

def create_distance_matrix(extracted_data, number):
    coordinates = []
    for node in extracted_data:
        if node['Cluster'] == number:
            coord = (node['CoordX'], node['CoordY'])
            coordinates.append(coord)
    print(coordinates)
    num_nodes = len(coordinates)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                coord_i = coordinates[i]
                coord_j = coordinates[j]
                distance = calculate_distance(coord_i, coord_j)
                distance_matrix[i][j] = distance

    return distance_matrix

def tsp_nearest_neighbor(coordinates, number):
    num_nodes = len(coordinates)
    visited = [False] * num_nodes

    # Wybierz losowy punkt początkowy
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

    # Dodaj odległość powrotną do punktu początkowego
    if number == -1:
        last_node = path[-1]
        distance_to_start = calculate_distance(coordinates[last_node], coordinates[start_node])
        total_distance += distance_to_start
        path.append(start_node)

    path_global = []

    for node in path:
        format_node = str(number) + "." + str(node + 1)
        path_global.append(format_node)

    if number >= 0:
        return path_global, total_distance, number, coordinates[0], coordinates[path[-1]]
    else:
        return path, total_distance, number, coordinates[0], coordinates[0]
    
def tsp_nearest_neighbor_global(coordinates, number):
    num_nodes = len(coordinates)
    visited = [False] * num_nodes

    # Wybierz losowy punkt początkowy
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

    # Dodaj odległość powrotną do punktu początkowego
    if number == -1:
        last_node = path[-1]
        distance_to_start = calculate_distance(coordinates[last_node], coordinates[start_node])
        total_distance += distance_to_start
        path.append(start_node)

    path_global = []

    for node in path:
        format_node = str(number) + "." + str(node + 1)
        path_global.append(format_node)

    if number >= 0:
        return path_global, total_distance, number, coordinates[0], coordinates[path[-1]]
    else:
        return path, total_distance, number, coordinates[0], coordinates[0]

def find_path(extracted_data):
    paths = []
    cluster_number = -1
    for node in extracted_data:
        if(node['Cluster'] > cluster_number):
            cluster_number = node['Cluster']
            cluster_coordinates = find_coordinates(extracted_data, cluster_number)
            if node['Cluster'] != 0:
                cluster_path = tsp_nearest_neighbor(cluster_coordinates, cluster_number)
                paths.append(cluster_path)
            else:
                paths.append([[0], 0, 0, cluster_coordinates[0], cluster_coordinates[0]])
    base_coordinates = []
    distance = 0
    for path in paths:
        distance += path[1]
        base_coordinates.append([path[3], path[2]])
        base_coordinates.append([path[4], path[2]])

    base_result = tsp_nearest_neighbor(base_coordinates, -1)
    base_path = base_result[0]
    print(base_path)
    distance += base_result[1]
    print("Distance:", distance)
    path = []
    for node in base_path:
        path.append(paths[node][0])
    print("Path:", path)
        
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

# Ścieżka do pliku XML
xml_file_path = 'dataset.xml'

# Wywołanie funkcji extract_data_from_xml
extracted_data = extract_data_from_xml(xml_file_path)

find_path(extracted_data)








