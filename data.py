import os
import json
import pandas as pd


def data_load(path):
    with open(path, "r") as file:
        data = json.load(file)

    return data


def Logging(output, save_path):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    res_path = os.path.join(save_path, 'Output.xlsx')

    res = pd.DataFrame(output)
    res.to_excel(res_path)


# {"paragraphs" : , "problems" :, "answers" :, "parag_num" : , "prob_num" : },