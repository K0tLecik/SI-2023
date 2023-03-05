import numpy as np
import pandas as pd
import csv
import math

# 3)
# loading chosen system
houses = np.loadtxt('house-votes-84.txt', dtype=str)
houses_types = np.loadtxt('house-votes-84-type.txt', dtype=str)

# a)
# available decision classes
adc = np.unique(houses[:, 16])
print(adc)

# b)
counter = [0] * len(adc)

for i in range(len(adc)):
    for j in houses[:, 16]:
        if adc[i] == j:
            counter[i] += 1
    print(adc[i], "=", counter[i])

# c)
# min and max values for each attribute
# atrybuty str

# d)
# number of different values for each attribute
for i in range(16):
    print(i, len(np.unique(houses[:, i])))

# e)
# list of all different values for each attribute
for i in range(16):
    print(i, np.unique(houses[:, i]))

# f)
# standard deviation for each attribute in the whole system and separately for each decision class
# atrybuty str

# 4)
# a)
df = pd.read_csv('Churn_Modelling.csv')

col_count = df.shape[0]
miss_col = int(col_count * 0.1)
random_miss_col = np.random.choice(col_count, size=miss_col, replace=False)

for col in df.columns:
    df.loc[random_miss_col, col] = np.nan

    if df[col].dtype == 'float64':
        atrybut_numeryczny = df[col].mean()
        df[col].fillna(atrybut_numeryczny, inplace=True)
    else:
        atrybut_symboliczny = df[col].mode()[0]
        df[col].fillna(atrybut_symboliczny, inplace=True)
print(df.head(10))

# b)
def normalize_attribute_value(path, index, interval_start, interval_end):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data.append(header)
        min_val = float('inf')
        max_val = float('-inf')
        for row in reader:
            attribute_value = float(row[index])
            if attribute_value < min_val:
                min_val = attribute_value
            if attribute_value > max_val:
                max_val = attribute_value
            data.append(row)

    normalized_data = []
    normalized_data.append(header)
    for row in data[1:]:
        attribute_value = float(row[index])
        normalized_value = ((attribute_value - min_val) * (interval_end - interval_start) / (
            max_val - min_val)) + interval_start
        row[index] = normalized_value
        normalized_data.append(row)

    with open('znormalizowane_dane.csv', 'w', newline='') as file:
        save = csv.writer(file)
        save.writerows(normalized_data)

    print(
        f"Atrybut  {header[index]} znormalizowany do przedziału ({interval_start}, {interval_end})")


normalize_attribute_value('Churn_Modelling.csv', 1, -1, 1)  # interval <-1,1>

# c)
def standarize(path, index):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)

        num_index = []
        for i, col in enumerate(header):
            if i != index and all(isinstance(row[i], (int, float)) for row in data):
                num_index.append(i)

        avg = [sum(float(row[i]) for row in data) / len(data) for i in num_index]
        standard_dev = [math.sqrt(sum((float(row[i]) - avg[j]) ** 2 for row in data) / len(data)) for j, i in
                    enumerate(num_index)]

        for i, row in enumerate(data):
            for j, index in enumerate(num_index):
                row[index] = (float(row[index]) - avg[j]) / standard_dev[j]
            data[i] = row

    with open('zad4c.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Atrybut {header[index]} zestandaryzowany")


standarize('Churn_Modelling.csv', 4)

# d)
data = pd.read_csv('Churn_Modelling.csv')

value_exists = pd.get_dummies(data['Geography'], prefix='Geography')
data = pd.concat([data, value_exists], axis=1)

data = data.drop(columns=['Geography_Germany'])

print(data.head())
