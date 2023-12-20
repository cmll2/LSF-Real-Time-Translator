import comparison_lib as cl

# Get the arguments

csv_paths, names = cl.get_paths_and_names()

cl.compare_models(csv_paths, names)

