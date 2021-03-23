from math import log

def prob(data, attr):
    freq = {}
    for obs in data:
        value = obs[attr]
        if value in freq.keys():
            freq[value] += 1
        else:
            freq[value] = 1
    for attribute, count in freq.items():
        freq[attribute] = count / len(data)
    return freq


def entropy(data, attr):
    sum = 0
    p = prob(data, attr)
    for value, pr in p.items():
        sum += pr * log(pr, 2)
    return -1 * sum


def ig(data, attr):
    i = entropy(data, 'play')
    ires = 0
    freq = {}
    for obs in data:
        if obs[attr] in freq.keys():
            freq[obs[attr]] += 1
        else:
            freq[obs[attr]] = 1

    for val in freq.keys():
        p = freq[val] / sum(freq.values())
        val_data = [record for record in data if record[attr] == val]
        ires += p * entropy(val_data, 'play')
    return i - ires


def get_iv(data, attr):
    iv = 0
    vals = [obs[attr] for obs in data]
    unique_vals = set(vals)

    for val in unique_vals:
        p = vals.count(val) / len(data)
        iv -= p * log(p, 2)
    return iv


def gr(data, attr):
    return ig(data, attr) / get_iv(data, attr)

def gini(data, attr):
    s = 0
    vals = [obs[attr] for obs in data]
    unique_vals = set(vals)

    for val in unique_vals:
        p = vals.count(val) / len(data)
        s += p**2
    return 1-s


def same_val(data):
    u = [record['play'] for record in data]
    return len(set(u)) == 1


def maximum_value(data, target_attr):
    vals = [record[target_attr] for record in data]
    count = {}
    for item in vals:
        if item in count.keys():
            count[item] += 1
        else:
            count[item] = 1
    return max(count.values())


def best_attribute(data, attr, crit):
    Gains = {}
    for i in attr:
        if crit == 'ig':
            Gains[i] = ig(data, i)
        elif crit == 'gr':
            Gains[i] = gr(data, i)
        else:
            Gains[i] = gini(data, i)

    maxGain = max(Gains.values())
    for attribute, attrGain in Gains.items():
        print(attribute+': '+str(attrGain))
        if attrGain == maxGain:
            maxAttr = attribute
    return maxAttr


def create_tree(data, attr, crit):
    tree = {}
    if same_val(data):
        return data[0]['play']
    else:
        A = best_attribute(data, attr, crit)
        tree = {A: {}}
        values = list(set([record[A] for record in data]))
        for val in values:
            examples = [rec for rec in data if rec[A] == val]
            tree[A][val] = create_tree(examples, [a for a in attr if a != A], crit)

    return tree


if __name__ == "__main__":
    data = [{'outlook': 'sunny', 'temp': '>=80', 'hum': '>=85', 'wind': False, 'play': False},
            {'outlook': 'sunny', 'temp': '>=80', 'hum': '>=85', 'wind': True, 'play': False},
            {'outlook': 'overcast', 'temp': '>=80', 'hum': '>=85', 'wind': False, 'play': True},
            {'outlook': 'rainy', 'temp': '70-79', 'hum': '>=85', 'wind': False, 'play': True},
            {'outlook': 'rainy', 'temp': '<70', 'hum': '75-84', 'wind': False, 'play': True},
            {'outlook': 'rainy', 'temp': '<70', 'hum': '<75', 'wind': True, 'play': False},
            {'outlook': 'overcast', 'temp': '<70', 'hum': '<75', 'wind': True, 'play': True},
            {'outlook': 'sunny', 'temp': '70-79', 'hum': '>=85', 'wind': False, 'play': False},
            {'outlook': 'sunny', 'temp': '<70', 'hum': '<75', 'wind': False, 'play': True},
            {'outlook': 'rainy', 'temp': '70-79', 'hum': '75-84', 'wind': False, 'play': True},
            {'outlook': 'sunny', 'temp': '70-79', 'hum': '<75', 'wind': True, 'play': True},
            {'outlook': 'overcast', 'temp': '70-79', 'hum': '>=85', 'wind': True, 'play': True},
            {'outlook': 'overcast', 'temp': '>=80', 'hum': '75-84', 'wind': False, 'play': True},
            {'outlook': 'rainy', 'temp': '70-79', 'hum': '>=85', 'wind': True, 'play': False},
            ]
    test = {'outlook': 'overcast', 'temp': '<70', 'hum': '<75', 'wind': False}
    predictors = ['outlook', 'temp', 'hum', 'wind']

    print('Selection Criteria: Information Gain')
    DecisionTree_ig = create_tree(data, predictors, 'ig')
    print(DecisionTree_ig)

    print('\nSelection Criteria: Gain Ratio')
    DecisionTree_gr = create_tree(data, predictors, 'gr')
    print(DecisionTree_gr)

    print('\nSelection Criteria: Gini Index')
    DecisionTree_gini = create_tree(data, predictors, 'gini')
    print(DecisionTree_gini)
