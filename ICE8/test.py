

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SNAdata = pd.read_csv("ICE8_Data.csv", index_col = 0)
G = nx.Graph(SNAdata)
nx.draw(G, with_labels = True)

plt.show()