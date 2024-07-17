from ssfm.ssfm.training_data_pipe import full_pipe
import pandas as pd
import concurrent.futures
#path = ''
#master_df = pd.read_csv('data_qual/master_delta_ratio.csv')
tup_list = []

for row in master_df.itertuples(): 
    tup_list.append((row[1], row[2], row[3], row[4]))
for t in tup_list:
    if t[0] == 'ACR_33':
        tup_list.remove(t)


def meta_pipe(tup):
    subject, recording, probe, channel = tup
    try:
        full_pipe(subject, recording, probe, channel)
    except Exception as e:
        ers = str(e)
        with open(f'td_fails/{subject}--{recording}--{probe}{channel}.txt', 'a') as f:
            f.write(f'{ers}')

with concurrent.futures.ProcessPoolExecutor(max_workers=224) as executor:
    # Map the function to the list of tuples
    executor.map(meta_pipe, tup_list)