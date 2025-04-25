import gym
import furniture
import pickle
import time

# Path to your demo file
demo_file = r"C:\Users\harsh\OneDrive - Georgia Institute of Technology\Documents\Georgia Tech\Spring 2025\CS 7648 IRL\Project\cs7648\demos\grasp_harsh_3\Sawyer_toy_table_0005.pkl"
txt_output = r"demos\demo_summary1.txt"

# Load demo data
with open(demo_file, "rb") as f:
    demo_data = pickle.load(f)
    # print(demo_data)
    
# --- extract info ---
keys = list(demo_data.keys())
actions = demo_data.get("actions", [])
n_actions = len(actions)
first_action = actions[0] if n_actions>0 else None
    
# --- write everything to txt ---
with open(txt_output, "w") as out:
    out.write(f"Demo file: {demo_file}\n\n")

    out.write(f"Keys in demo_data: {keys}\n\n")
    out.write(f"Number of actions: {n_actions}\n")
    out.write("All actions:\n")
    for i, act in enumerate(actions):
        out.write(f"{i:04d}: {act.tolist()}\n")

    for key, val in demo_data.items():
        out.write(f"=== {key} ===\n")
        # if it's a list, print length and then each entry
        if isinstance(val, list):
            out.write(f"Length: {len(val)}\n")
            for i, entry in enumerate(val):
                # for numpy arrays or dicts we cast to list or str
                if hasattr(entry, "tolist"):
                    entry_str = entry.tolist()
                else:
                    entry_str = entry
                out.write(f"{i:04d}: {entry_str}\n")
        # if it's a numpy array (unlikely top‐level), dump shape + contents
        elif hasattr(val, "shape"):
            out.write(f"Array shape: {val.shape}\n")
            out.write(f"{val.tolist()}\n")
        # metadata might be a dict
        elif isinstance(val, dict):
            for mk, mv in val.items():
                out.write(f"{mk}: {mv}\n")
        else:
            out.write(f"{val}\n")
        out.write("\n")

print(f"Saved full summary to {txt_output}")

# Print some basic info about the demo
print("Keys in demo_data:", demo_data.keys())

# Get the actions recorded in the demo
actions = demo_data.get("actions", None)
if actions is None:
    print("No actions found in demo data.")
    exit()