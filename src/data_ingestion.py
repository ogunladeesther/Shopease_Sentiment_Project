import pandas as pd
from pathlib import Path
import re
import numpy as np
import logging
from Config.constant import input_data

logging.basicConfig(level=logging.INFO)



def data_ingestion():
    try:
        data = pd.read_csv(input_data)
        logging.info("Data successfully loaded .....")
        print(data.head())
        return data
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
