import os
import pandas as pd
from config.options import TASKLIST
from tqdm import tqdm


def select(file_to_modify, target_file, output_file):
    id_list = pd.read_csv(target_file)
    data = pd.read_csv(file_to_modify)
    id_list = id_list[['ID', *TASKLIST]]
    # print duplicate ID in data
    # print("duplicate ID in data:", data['ID'][data['ID'].duplicated()])

    try:
        select_data = match_id(data, id_list)
    except AssertionError as e:
         raise AssertionError(e)
    

    # save the selected data
    select_data.to_csv(os.path.join(output_file), index=False)


def match_id(source_df, target_df):
    data = source_df
    id_list = target_df['ID'].tolist()
    # select the rows that ID is in id_list
    select_data = data[data['ID'].isin(target_df['ID'])]

    # only keep the columns in id_list for the selected data
    select_data = select_data[target_df.columns]

    # # remove duplicate ID in select_data
    # if len(select_data['ID']) != len(target_df['ID']):
    #     select_data = select_data.drop_duplicates(subset=['ID'])

    if not source_df['ID'].equals(target_df['ID']):
        # re-order the rows, make selected data the same as id_list
        new_data = []
        for id in tqdm(id_list, desc="Matching ID"):
            row = select_data[select_data['ID'] == id]
            assert len(row) >0, f"ID {id} not found in select_data"
            new_data.append(row.to_dict(orient='records')[0])
        select_data = pd.DataFrame(new_data)

    if not source_df['ID'].equals(target_df['ID']):
        print('ID not match after selection, finding the missing ID')
        # make sure id matches, output missing ID
        for i, each in enumerate(target_df["ID"].tolist()):
                assert select_data["ID"].tolist()[i] == each, f"ID does not match in {each}"
    return select_data
    # f"ID does not match in {file_to_modify}"
