import pandas as pd
from sklearn import svm
import numpy as np
from sklearn import svm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

seed = 10599
aqua_globals.random_seed = seed

df = pd.read_csv("Trial_Data.zip")

selection = ["Tanker", "Cargo"]
df["shiptype"] = df["shiptype"].replace(np.nan, "nan")

# for ind in df.index:
#     name = df["shiptype"][ind]
#     if name not in selection and name != "nan":
#         df["shiptype"][ind] = "Rest"

mask = df.shiptype.apply(lambda x: any(item for item in selection if item in x))
df = df[mask]
unique_ships = df.mmsi.unique()
ship_classes = df.shiptype
df = df.drop("shiptype", axis=1)
df = df.drop("cog", axis=1)
df = df.drop("navigationalstatus", axis=1)
df.fillna(df.mean(), inplace=True)
df.shape
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
df = df*2*np.pi
df.shape
X = df
y = ship_classes
kpca = PCA(n_components=5, whiten=True)
# 5 is good
principalComponents = kpca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    principalComponents, y, test_size=15000, random_state=0
)

X_train.shape
mask1 = y_train=="Cargo"
mask2 = y_train=="Tanker"
train_inp = {"Cargo": X_train[mask1][:100],
            "Tanker": X_train[mask2][:100]}
mask1 = y_test=="Cargo"
mask2 = y_test=="Tanker"
test_inp = {"Cargo": X_test[mask1][:20],
            "Tanker": X_test[mask2][:20]}
feature_dim = 5
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='circular')
qsvm = QSVM(feature_map, train_inp, test_inp)

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

result = qsvm.run(quantum_instance)

print(f'Testing success ratio: {result["testing_accuracy"]}')