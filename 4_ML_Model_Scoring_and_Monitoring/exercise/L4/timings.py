
import os
import timeit
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Function for ingestion timing
def ingestion_timing():
    start = timeit.default_timer()
    os.system("python3 ingestion.py")
    stop = timeit.default_timer()
    ingestion_time_taken = stop - start
    return ingestion_time_taken

# Function for training timing
def training_timing():
    start = timeit.default_timer()
    os.system("python3 training.py")
    stop = timeit.default_timer()
    training_time_taken = stop - start
    return training_time_taken

# Function for measure timings
def measure_and_save_timings():
    ingestion_times = []
    training_times = []
    for i in range(20):
        ingestion_times.append(ingestion_timing())
        training_times.append(training_timing())
    
    # calculate mean, std, min, max
    ingestion_mean = np.mean(ingestion_times)
    ingestion_std = np.std(ingestion_times)
    ingestion_min = np.min(ingestion_times)
    ingestion_max = np.max(ingestion_times)

    training_mean = np.mean(training_times)
    training_std = np.std(training_times)
    training_min = np.min(training_times)
    training_max = np.max(training_times)
    
    logging.info(f"Ingestion mean: {ingestion_mean} std: {ingestion_std} min: {ingestion_min} max: {ingestion_max}")
    logging.info(f"Training mean: {training_mean} std: {training_std} min: {training_min} max: {training_max}")

    results= {'ingestion_mean': ingestion_mean, 'ingestion_std': ingestion_std, 'ingestion_min': ingestion_min, 'ingestion_max': ingestion_max, 'training_mean': training_mean, 'training_std': training_std, 'training_min': training_min, 'training_max': training_max}

    return results

if __name__ == '__main__':
    results = measure_and_save_timings()
    logging.info(f"Results: {results}")