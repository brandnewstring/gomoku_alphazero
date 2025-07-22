import os
from datetime import date

# general game and model params
board_size = 11  # yes it's square, no point making length and width different IMO.
draw_threshold = 0.7 # if 70% of the board is filled, treat it as a draw.
date_str = date.today().strftime("%y_%m_%d")
# folder_name = f"pvnet_{board_size}x{board_size}_{date_str}"
folder_name = f"pvnet_{board_size}x{board_size}_2nd_model"
os.makedirs(folder_name, exist_ok=True)
current_model = f"{folder_name}/current_policy.keras"
best_model = f"{folder_name}/best_policy.keras"