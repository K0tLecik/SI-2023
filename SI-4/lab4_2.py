import itertools


#zad1
def check_consistency(rules, decision_system):
    for rule in rules:
        for row in decision_system:
            if all(rule[i] == row[i] for i in rule):
                if rule[-1] != row[-1]:
                    return False
    return True

def find_rules(decision_system):
    attributes = len(decision_system[0]) - 1
    rules = []

    while decision_system:
        found = False
        for row in decision_system:
            for length in range(1, attributes + 1):
                if found:
                    break
                for combination in itertools.combinations(range(attributes), length):
                    rule = {i: row[i] for i in combination}
                    rule[-1] = row[-1]
                    if check_consistency([rule], decision_system):
                        rules.append(rule)
                        decision_system = [r for r in decision_system if not all(rule[i] == r[i] for i in rule)]
                        found = True
                        break

    return rules

decision_system = [
    [1, 1, 1, 1, 3, 1, 1],
    [1, 1, 1, 1, 3, 2, 1],
    [1, 1, 1, 3, 2, 1, 0],
    [1, 1, 1, 3, 3, 2, 1],
    [1, 1, 2, 1, 2, 1, 0],
    [1, 1, 2, 1, 2, 2, 1],
    [1, 1, 2, 2, 3, 1, 0],
    [1, 1, 2, 2, 4, 1, 1]
]

rules = find_rules(decision_system)
print("Reguły:")
for rule in rules:
    print(rule)
