"""Regenerates only the two network diagrams (fast; no cross-validation).
Run from the project root:  python draw_networks.py"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from evaluate import learn_structure, draw_network, EDGES, EXPERT_POS, DATA, COLS

df = pd.read_csv(DATA)[COLS].astype(str)
draw_network(EDGES, "Expert-designed structure", "model/network_expert.png", EXPERT_POS)
learned = learn_structure(df)
print("Learned edges:", learned)
draw_network(learned, "Learned structure (hill climbing + BIC)", "model/network_learned.png")
print("Saved model/network_expert.png and model/network_learned.png")
