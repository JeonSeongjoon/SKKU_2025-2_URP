import json
import pandas as pd


def data_load(path):
    data = json.load(path)
    return data


def toExcel(output):
    res = pd.DataFrame(output)
    res.to_excel('Output')
