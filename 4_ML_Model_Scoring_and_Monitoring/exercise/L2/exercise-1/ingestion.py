import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

# read only csv and json from data directories
def read_data(dirs):
    total_files = 0
    data = pd.DataFrame()
    for dir in dirs:
        logging.info(f"Reading data from {dir}")
        for file in os.listdir(dir):
            logging.info(f"Reading {file}")
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(dir, file))
                data = data.append(df)
                total_files += 1
            elif file.endswith('.json'):
                df = pd.read_json(os.path.join(dir, file), lines=True)
                data = data.append(df)
                total_files += 1

    # remove duplicates
    logging.info(f"Data shape before cleaning: {data.shape}")
    data.drop_duplicates(inplace=True)
    logging.info(f"Read {total_files} files")
    # save cleaned data to csv
    data.to_csv('result.csv', index=False)
    logging.info(f"Data shape after cleaning: {data.shape}")
    logging.info(f"Saved cleaned data to result.csv")
    return data, total_files

if __name__ == '__main__':
    dirs = ['data1', 'data2', 'data3']
    data, total_files = read_data(dirs)
