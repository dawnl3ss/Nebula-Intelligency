import numpy as np
from Data.DataEntry import DataEntry
from System.NeuronalSystem import NeuronalSystem

x_entry = np.array(
    DataEntry((
        [3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4.5, 1.5]
    )).get_data(), dtype = float
)

y_entry = np.array(
    DataEntry((
        [1], [0], [1], [0], [1], [0], [1], [0]
    )).get_data(), dtype = float
)

x_entry = x_entry / np.amax(x_entry, axis = 0)

search = np.split(x_entry, [8])[0]
prediction = np.split(x_entry, [8])[1]

sys = NeuronalSystem()
output = sys.ia_forward(search)

for i in range(1000):
    print("# " + str(i))
    print("Valeurs d'entre : \n" + str(search))
    print("Sortie actuelle : \n" + str(y_entry))
    print("Sortie predite : \n" + str(np.matrix.round(sys.ia_forward(search), 2)))
    sys.handle(search, y_entry)