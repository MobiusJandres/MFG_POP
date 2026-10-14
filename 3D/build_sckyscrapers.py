# %%

import sys
import gmsh

# 1. Initialize Gmsh

gmsh.initialize() 
#This is basically the environment for where the objects will set. 
gmsh.model.add("skyscraper_swarm_domain")

# Use OpenCASCADE kernel for 3D constructive solid geometry
occ = gmsh.model.occ

# 2. Define overall air domain box (x, y, z, dx, dy, dz)
domain_x, domain_y, domain_z = 100.0, 100.0, 50.0
air_box = occ.addBox(0, 0, 0, domain_x, domain_y, domain_z)

# 3. Create 4 skyscraper building boxes
# Format: addBox(x_origin, y_origin, z_origin, length, width, height)
b1 = occ.addBox(15, 15, 0, 20, 20, 35)  # Building 1: 35m tall
b2 = occ.addBox(60, 15, 0, 25, 15, 40)  # Building 2: 40m tall
b3 = occ.addBox(15, 60, 0, 15, 25, 30)  # Building 3: 30m tall
b4 = occ.addBox(55, 55, 0, 25, 25, 45)  # Building 4: 45m tall

# 4. Subtract buildings from the surrounding air box
# Dimension 3 corresponds to volumetric shapes
air_domain, _ = occ.cut([(3, air_box)], [(3, b1), (3, b2), (3, b3), (3, b4)])

# Synchronize OpenCASCADE representations with Gmsh internal model
occ.synchronize()

# 5. Control Mesh Element Sizing
# Finer mesh near obstacles and coarse mesh in open air
gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 8.0)

# 6. Tag Physical Groups for PDE / MFG Solvers
# Volume tag for navigable air domain
vol_entities = gmsh.model.getEntities(dim=3)
vol_tags = [e[1] for e in vol_entities]
gmsh.model.addPhysicalGroup(3, vol_tags, name="navigable_air")

# Surface tags for obstacle boundaries
surfaces = gmsh.model.getEntities(dim=2)
surf_tags = [s[1] for s in surfaces]
gmsh.model.addPhysicalGroup(2, surf_tags, name="domain_boundaries")

# 7. Generate 3D Tetrahedral Mesh
gmsh.model.mesh.generate(3)

# 8. Export Mesh
gmsh.write("skyscraper_swarm_domain.msh")

# Launch visualizer GUI (close window to exit script)
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
# %%
