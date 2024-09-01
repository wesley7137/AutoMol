import pymol

# Initialize PyMOL
pymol.finish_launching()

# Load the structure
pymol.cmd.load(r"C:\Users\wes\vectordb_data_good\scripts\em3_predictions\round_tripped.pdb", "protein")

# Set visualization style
pymol.cmd.show("cartoon", "protein")
pymol.cmd.color("spectrum", "protein")

# Save the image
pymol.cmd.ray(1024, 768)
pymol.cmd.png("protein_structure.png", dpi=300)

# Close PyMOL
pymol.cmd.quit()
