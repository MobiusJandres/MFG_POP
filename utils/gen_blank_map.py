import numpy as np
from pathlib import Path

# Create goal directories
maps_dir = Path("Maps").absolute()
scen_dir = Path("Scenarios").absolute()
maps_dir.mkdir(parents=True, exist_ok=True)
scen_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate 768x768 Open Map
map_filename = "empty_768x768.map"
map_path = maps_dir / map_filename
map_header = "type octile\nheight 768\nwidth 768\nmap\n"
map_body = ("." * 768 + "\n") * 768

with open(map_path, "w") as f:
    f.write(map_header + map_body)

# 2. Define High-Density Crowd Cluster Centers (x, y)
cluster_centers = [
    (200, 200),  # Bottom-left crowd
    (568, 200),  # Bottom-right crowd
    (200, 568),  # Top-left crowd
    (568, 568),  # Top-right crowd
    (384, 384),  # Central crowd
]

agents_per_cluster = 50  # 5 clusters * 30 = 150 total agents
scen_entries = ["version 1.0"]

agent_id = 0
rng = np.random.default_rng(seed=42)  # Seed for reproducible placement

additional_centers = [ np.random.randint(0, 768,2) for count in range(20)]
cluster_centers = cluster_centers + additional_centers

# Interleave cluster agents so loading the first 50 agents populates all crowds
for i in range(agents_per_cluster):
    for cx, cy in cluster_centers:
        bucket = agent_id // 10
        # Place start coordinates tightly around cluster center (std dev = 10 cells)
        sx = int(np.clip(rng.normal(cx, 10.0), 10, 757))
        sy = int(np.clip(rng.normal(cy, 10.0), 10, 757))
        
        # Nominal goal coordinates
        gx, gy = 384, 384
        dist = np.sqrt((gx - sx)**2 + (gy - sy)**2)
        
        entry = f"{bucket}\t{map_filename}\t768\t768\t{sx}\t{sy}\t{gx}\t{gy}\t{dist:.8f}"
        scen_entries.append(entry)
        agent_id += 1

scen_filename = "empty_768x768.map.scen"
scen_path = scen_dir / scen_filename

with open(scen_path, "w") as f:
    f.write("\n".join(scen_entries) + "\n")

print(f"Generated obstacle-free map at: {map_path}")
print(f"Generated scenario file with {agent_id} agents across 5 high-density clusters at: {scen_path}")