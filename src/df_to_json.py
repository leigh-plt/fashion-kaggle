import pandas as pd
import numpy as np
import json, argparse

def convert():
    
    df = pd.read_csv(args.csv_file)
    
    mask = {}
    
    for _, row in df.iterrows():

        if '_' in row.ClassId:
            class_id = int(row.ClassId.split('_')[0])
            attr_id = row.ClassId.split('_')[1:]
        else:
            class_id = int(row.ClassId)
            attr_id = []

        if row.ImageId in mask:
            mask[row.ImageId]['mask'].append([class_id, row.EncodedPixels, attr_id])
        else:   
            mask[row.ImageId] = {'Width': row.Width,
                                'Height': row.Height,
                                'mask': []}
            mask[row.ImageId]['mask'].append([class_id, row.EncodedPixels, attr_id])

    with open(args.json_name, 'w') as file:
        json.dump(mask, file)
        
parser = argparse.ArgumentParser(description='')

parser.add_argument('--csv_file', default='data/train.csv', type=str, help='Input csv file')
parser.add_argument('--json_name', default='data/train.json', type=str, help='output filename for structured dict')

args = parser.parse_args()

if __name__ == '__main__':
    
    convert()