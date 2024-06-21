import os
import json
import pdb
import pandas as pd

explore_bs=False
# # Directory containing all the folders
n_shot="5"
data_name="gsm8k"
# main_dir = f'./result/{data_name}/{n_shot}_shot/'
# save_filename =  f'results_summary_{data_name}_{n_shot}shot.csv'
main_dir = f'./result/{data_name}/cot/'
save_filename =  f'results_summary_{data_name}_cot.csv'
explore_bs=False

# n_shot="25"
# data_name="arc_challenge"
# main_dir = f'./result/{data_name}/{n_shot}_shot/'
# save_filename =  f'results_summary_{data_name}_{n_shot}shot.csv'
# # main_dir = f'./result/{data_name}/shuffle_40/'
# # save_filename =  f'results_summary_{data_name}_shuffle.csv'
# explore_bs=False
# # main_dir = f'./result/{data_name}/batch_size/'
# # save_filename =  f'results_summary_{data_name}_bs_{n_shot}shot.csv'
# # explore_bs=True

# n_shot="0"
# data_name="truthfulqa"
# main_dir = f'./result/{data_name}/mc2/'
# save_filename =  f'results_summary_{data_name}_mc2_{n_shot}shot.csv'
# # main_dir = f'./result/{data_name}/shuffle'
# # save_filename =  f'results_summary_{data_name}_shuffle_{n_shot}shot.csv'
# # main_dir = f'./result/{data_name}/mc1'
# # save_filename =  f'results_summary_{data_name}_mc1_{n_shot}shot.csv'
# # main_dir = f'./result/{data_name}/mc2_sample_answer'
# # save_filename =  f'results_summary_{data_name}_mc2_sample_answer_{n_shot}shot.csv'
# main_dir = f'./result/{data_name}/mc2_sample_both'
# save_filename =  f'results_summary_{data_name}_mc2_sample_both_{n_shot}shot.csv'


# n_shot="7"
# data_name="csqa"
# main_dir = f'./result/{data_name}/{n_shot}_shot/'
# save_filename =  f'results_summary_{data_name}_{n_shot}shot.csv'
# main_dir = f'./result/{data_name}/gen/'
# save_filename =  f'results_summary_{data_name}_gen_{n_shot}shot.csv'
# main_dir = f'./result/{data_name}/shuffle/'
# save_filename =  f'results_summary_{data_name}_shuffle_{n_shot}shot.csv'
# main_dir = f'./result/{data_name}/gen_shuffle/'
# save_filename =  f'results_summary_{data_name}_gen_shuffle_{n_shot}shot.csv'


# Dictionary to hold the data
data = {}

# Iterate over each folder in the main directory
for folder in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder)
    if not os.path.isdir(folder_path): continue
    dir_contents = os.listdir(folder_path)
    if not dir_contents: continue # Check if the list is not empty
    parts = folder.split('_')
    model = parts[0]
    task = '_'.join(parts[1:-1])  # Task may contain underscores other than the ones in 'bs{batch_size}'
    batch_size = parts[-1]  # Assuming batch_size is the last part after the last underscore

    # Find the subfolder
    for subfolder in os.listdir(folder_path):
        if subfolder.startswith("cache"): continue
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            # Find the json file
            for file in os.listdir(subfolder_path):
                if file.startswith('results'):
                    json_path = os.path.join(subfolder_path, file)
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        # import pdb
                        # pdb.set_trace()
                        model_name = json_data['model_name']
                        if 'exact_match,flexible-extract' in json_data['results'][task]:
                            # gsm8k
                            number = json_data['results'][task]['exact_match,flexible-extract']
                        elif 'acc_norm,none' in json_data['results'][task]:
                            # arc
                            number = json_data['results'][task]['acc_norm,none']
                        elif 'acc,none' in json_data['results'][task]:
                            # truthfulqa
                            number = json_data['results'][task]['acc,none']
                        else:
                            # gsm8k-cot
                            number = json_data['results'][task]['exact_match,maj@8']
                            number_64 = json_data['results'][task].get('exact_match,maj@64', 0)
                        number = round(float(number)*100, 2)
                        if model_name not in data:
                            data[model_name] = {}
                        if explore_bs:
                            data[model_name][f"{task}_{batch_size}"] = number
                        elif 'exact_match,maj@8' in json_data['results'][task]:
                            number_64 = round(float(number_64)*100, 2)
                            data[model_name][f"{task}_maj@8"] = number
                            data[model_name][f"{task}_maj@64"] = number_64
                        else:
                            data[model_name][f"{task}"] = number

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data, orient='index').fillna('')

# Sort DataFrame by index (model_name)
df_sorted = df.sort_index()

# Save DataFrame to CSV
csv_path = os.path.join("./result/", save_filename)
df_sorted.to_csv(csv_path)
print(f"CSV file has been saved to {csv_path}")
