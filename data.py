import json
import pandas as pd


def data_load(path):
    with open(path, "r") as file:
        data = json.load(file)

    return data


def toExcel(output):
    res = pd.DataFrame(output)
    res.to_excel('Output')
