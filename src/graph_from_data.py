import numpy as np
import json
from pandas import DataFrame

from src.graph import DataFrameGraph

def get_facebook_dataframe_graph(cut = None):
    with open('data/facebook.edges') as file:
        lines = file.readlines()
        nodes_from = [''] * len(lines)
        nodes_to = [''] * len(lines)

        for i, line in enumerate(lines):
            nodes_from[i], nodes_to[i] = line.split()

        if cut is not None and cut > 0:
            nodes_from = nodes_from[:cut]
            nodes_to = nodes_to[:cut]

        df = DataFrame(
            {
                "from": nodes_from,
                "to": nodes_to
            }
        )
        return DataFrameGraph(df)

def get_deezer_dataframe_graph(idx:int = None, cut = None):
    with open('data/deezer_edges.json') as file:
        deezer_graphs = json.load(file)
        lines = deezer_graphs[str(idx)]
        nodes_from = [''] * len(lines)
        nodes_to = [''] * len(lines)

        for i, line in enumerate(lines):
            nodes_from[i], nodes_to[i] = str(line[0]), str(line[1])

        if cut is not None and cut > 0:
            nodes_from = nodes_from[:cut]
            nodes_to = nodes_to[:cut]

        df = DataFrame(
            {
                "from": nodes_from,
                "to": nodes_to
            }
        )
        return DataFrameGraph(df)