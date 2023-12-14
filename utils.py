import json

def save_stats(key, value, stats_path):    
    try:
        with open(stats_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    
    data[key] = value
    
    with open(stats_path, 'w') as file:
        json.dump(data, file)