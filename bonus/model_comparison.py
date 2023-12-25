# ------------------------------------------- Imports ---------------------------------------------------------------------------
print("Importing libraries...")
import comparison_lib as cl

# ------------------------------------------- Get the arguments from the command line -------------------------------------------
print("Getting arguments from command line...")
csv_paths, names = cl.get_paths_and_names()

# ------------------------------------------- Compare the models ----------------------------------------------------------------
print("Comparing models...")
cl.compare_models(csv_paths, names)

