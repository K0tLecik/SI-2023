import pandas as pd
import numpy as np
from functools import reduce

def get_discernibility_matrix(data):
    n = data.shape[0]
    discernibility_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                discernibility_matrix[i, j] = sum(data.iloc[i, :-1] != data.iloc[j, :-1])
    return discernibility_matrix

def get_all_reducts(decision_system):
    discernibility_matrix = get_discernibility_matrix(decision_system)
    n, m = decision_system.shape
    attributes = set(range(m - 1))
    reducts = set()

    def search_reduct(current_reduct, remaining_attributes):
        if np.all(reduce(np.bitwise_or, discernibility_matrix[:, list(current_reduct)])):
            reducts.add(frozenset(current_reduct))
        for attr in remaining_attributes:
            search_reduct(current_reduct | {attr}, remaining_attributes - {attr})

    search_reduct(set(), attributes)
    return [set(reduct) for reduct in reducts]

def get_all_rules(decision_system, reducts):
    all_rules = []
    for reduct in reducts:
        unique_rows = decision_system.iloc[:, list(reduct) + [-1]].drop_duplicates()
        rules = []
        for _, row in unique_rows.iterrows():
            conditions = [f"{decision_system.columns[col]} = {row.at[decision_system.columns[col]]}" for col in reduct]
            rule = f"Jeśli {' i '.join(conditions)}, to dec = {row['dec']}"
            rules.append(rule)
        all_rules.append(rules)
    return all_rules

# Zadanie 1
print("\nZadanie 1")
decision_system1 = pd.DataFrame({
    'a': [0, 1, 2, 0],
    'b': [2, 2, 0, 2],
    'c': [1, 2, 2, 1],
    'd': [0, 1, 1, 1],
    'dec': [0, 0, 1, 2]
})

reducts1 = get_all_reducts(decision_system1)
print(f"Wszystkie redukty decyzyjne dla Fig. 1: {reducts1}")

# Zadanie 2
print("\nZadanie 2")
all_rules1 = get_all_rules(decision_system1, reducts1)
print("\nWszystkie reguły wygenerowane z otrzymanych reduktów decyzyjnych:")
for i, rules in enumerate(all_rules1):
    print(f"Redukt {i + 1}:")
    for rule in rules:
        print(f"  {rule}")
# Zadanie 3
print("\nZadanie 3")
decision_system2 = pd.DataFrame({
    'a1': ['wysoka', 'wysoka', 'wysoka', 'więcej niż średnia', 'więcej niż średnia', 'więcej niż średnia', 'wysoka', 'więcej niż średnia', 'więcej niż średnia'],
    'a2': ['bliski', 'bliski', 'bliski', 'daleki', 'daleki', 'daleki', 'bliski', 'daleki', 'daleki'],
    'a3': ['średni', 'średni', 'średni', 'silny', 'silny', 'lekki', 'średni', 'lekki', 'lekki'],
    'dec': ['tak', 'tak', 'tak', 'nie pewne', 'nie', 'nie', 'tak', 'nie', 'tak']
})

X1, X2 = 'a1', 'a2'
A = {X2}
B = {X1, X2}

subset_A = decision_system2.loc[:, list(A) + ['dec']].drop_duplicates()
subset_B = decision_system2.loc[:, list(B) + ['dec']].drop_duplicates()

print("Opis dla X2 w odniesieniu do A:")
grouped_A = subset_A.groupby(X2)["dec"].unique()
for a2_value, dec_values in grouped_A.items():
    print(f"Jeśli {X2} = {a2_value}, to dec =", ', '.join(dec_values))

print("Opis dla X1 i X2 w odniesieniu do B:")
for _, row in subset_B.iterrows():
    print(f"Jeśli {X1} = {row[X1]} i {X2} = {row[X2]}, to dec = {row['dec']}")
    print("\nZadanie 4")
    reduct2 = get_all_reducts(decision_system2)
    print(f"Redukt decyzyjny dla Fig. 2: {reduct2}")

    rules2 = get_all_rules(decision_system2, reduct2)
    print("Reguły wygenerowane z otrzymanego reduktu decyzyjnego dla Fig. 2:")
    for rule in rules2:
        print(rule)
