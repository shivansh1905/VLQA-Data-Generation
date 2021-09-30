import json
import random

with open('annotations.json') as json_file:
    data = json.load(json_file)

#with open('variable_names.txt') as vnames:
    #list_color = json.load(vnames)

def data_gen(num):
    dict_json = {}
    for i in range(num):
        current = random.choice(data)
        data.remove(current)

        if current['type'] == 'hbar_categorical':
            colors_available = []
            for k in range(len(current['models'][0]['labels'])):
                color = current['models'][0]['labels'][k]
                colors_available.append(color)

            to_be_changed = random.choice(colors_available)
            dict_json['Entry_{}'.format(i)] = []
            dict_json['Entry_{}'.format(i)].append("dataset_id: figureqa-train1-v1")
            dict_json['Entry_{}'.format(i)].append("Image_index: {}".format(current['image_index']))
            dict_json['Entry_{}'.format(i)].append("Graph_type: {}".format(current['type']))
            dict_json['Entry_{}'.format(i)].append("Passage: {}'s value is reduced by 10 %.".format(to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Question: What is the new value of {}, after being changed?".format(to_be_changed))
            answer_hbar = round(current['models'][0]['x'][colors_available.index(to_be_changed)] * 0.9, 1)
            dict_json['Entry_{}'.format(i)].append("Answer: The new value of {} would be {}".format(to_be_changed, answer_hbar))

        if current['type'] == 'vbar_categorical':
            colors_available = list()
            for k in range(len(current['models'][0]['labels'])):
                color = current['models'][0]['labels'][k]
                colors_available.append(color)

            to_be_changed = random.choice(colors_available)
            dict_json['Entry_{}'.format(i)] = []
            dict_json['Entry_{}'.format(i)].append("dataset_id: figureqa-train1-v1")
            dict_json['Entry_{}'.format(i)].append("Image_index: {}".format(current['image_index']))
            dict_json['Entry_{}'.format(i)].append("Graph_type: {}".format(current['type']))
            dict_json['Entry_{}'.format(i)].append("Passage: {}'s value is reduced by 25 %.".format(to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Question: What is the new value of {}, after being changed?".format(to_be_changed))
            answer_vbar = round(current['models'][0]['y'][colors_available.index(to_be_changed)] * 0.75, 1)
            dict_json['Entry_{}'.format(i)].append("Answer: The new value of {} would be {}".format(to_be_changed, answer_vbar))

        if current['type'] == 'pie':
            colors_available, spans = list(), list()
            for j in range(len(current['models'])):
                color = current['models'][j]['name']
                colors_available.append(color)
            for k in range(len(current['models'])):
                span = current['models'][k]['span']
                spans.append(span)

            to_be_changed = random.choice(colors_available)
            to_compare_with = random.choice([color for color in colors_available if color != to_be_changed])
            index_of_to_be_changed = colors_available.index(to_be_changed)
            index_of_to_compare_with = colors_available.index(to_compare_with)
            span_of_to_be_changed = spans[index_of_to_be_changed]
            span_of_to_compare_with = spans[index_of_to_compare_with]

            dict_json['Entry_{}'.format(i)] = []
            dict_json['Entry_{}'.format(i)].append("dataset_id: figureqa-train1-v1")
            dict_json['Entry_{}'.format(i)].append("Image_index: {}".format(current['image_index']))
            dict_json['Entry_{}'.format(i)].append("Graph_type: {}".format(current['type']))

            dict_json['Entry_{}'.format(i)].append("Passage: The portion of {} is dropped by 30 % of its value. ".format(to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Question: What would now be the percentage difference between new {} value and {} ? ".format(to_be_changed, to_compare_with))

            new_span_for_to_be_changed = 0.7 * (span_of_to_be_changed)
            answer_pie = round((new_span_for_to_be_changed / (span_of_to_be_changed+span_of_to_compare_with))*100, 1)
            dict_json['Entry_{}'.format(i)].append("Answer: The percentage difference between new {} value and {} is {} %.".format(to_be_changed, to_compare_with,answer_pie))

        if current['type'] == 'line':
            colors_available, y_values = list(), list()
            for j in range(len(current['models'])):
                color = current['models'][j]['name']
                colors_available.append(color)
            to_be_changed = random.choice(colors_available)
            to_be_derived = random.choice([color for color in colors_available if color != to_be_changed])

            dict_json['Entry_{}'.format(i)] = []
            dict_json['Entry_{}'.format(i)].append("dataset_id: figureqa-train1-v1")
            dict_json['Entry_{}'.format(i)].append("Image_index: {}".format(current['image_index']))
            dict_json['Entry_{}'.format(i)].append("Graph_type: {}".format(current['type']))

            for k in range(len(current['models'])):
                y_value = current['models'][k]['y'][-1]  # y-values at max-x
                y_values.append(y_value)

            index_of_to_be_changed = colors_available.index(to_be_changed)
            index_of_to_be_derived = colors_available.index(to_be_derived)
            y_val_of_to_be_changed = y_values[index_of_to_be_changed]
            y_val_of_to_be_derived = y_values[index_of_to_be_derived]

            new_y_value_to_be_changed = round((3 * y_val_of_to_be_derived) + y_val_of_to_be_changed, 1)

            dict_json['Entry_{}'.format(i)].append("Passage: Thrice of {}: y value is added to {}: y value for the maximum x-value.".format(to_be_derived, to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Question: What would be the new y value of {} after the change? ".format(to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Answer: It would be {}".format(new_y_value_to_be_changed))


        if current['type'] == 'dot_line':
            colors_available = list()
            for j in range(len(current['models'])):
                color = current['models'][j]['name']
                colors_available.append(color)

            to_be_changed = random.choice(colors_available)
            to_be_derived = random.choice([color for color in colors_available if color != to_be_changed])

            y_values_at_maxX = list()
            for k in range(len(current['models'])):
                y_value = current['models'][k]['y'][-1]  # y-values at max-x
                y_values_at_maxX.append(y_value)

            y_values_at_minX = list()
            for m in range(len(current['models'])):
                y_value = current['models'][m]['y'][0]
                y_values_at_minX.append(y_value)

            index_of_to_be_changed = colors_available.index(to_be_changed)
            index_of_to_be_derived = colors_available.index(to_be_derived)
            y_val_of_to_be_changed_maxX = y_values_at_maxX[index_of_to_be_changed]
            y_val_of_to_be_derived_maxX = y_values_at_maxX[index_of_to_be_derived]
            y_val_of_to_be_derived_minX = y_values_at_minX[index_of_to_be_derived]


            dict_json['Entry_{}'.format(i)] = []
            dict_json['Entry_{}'.format(i)].append("dataset_id: figureqa-train1-v1")
            dict_json['Entry_{}'.format(i)].append("Image_index: {}".format(current['image_index']))
            dict_json['Entry_{}'.format(i)].append("Graph_type: {}".format(current['type']))

            dict_json['Entry_{}'.format(i)].append("Passage: Twice of {} at minimum X-value is added to {} at maximum X-value".format(to_be_derived,to_be_changed))
            dict_json['Entry_{}'.format(i)].append("Question: What is the absolute difference between new {} and {} at maximum X-value ?".format(to_be_changed, to_be_derived))

            summation = (y_val_of_to_be_changed_maxX + (2 * y_val_of_to_be_derived_minX))
            answer_dot_line = round(abs(summation - y_val_of_to_be_derived_maxX), 1)

            dict_json['Entry_{}'.format(i)].append("Answer: It would be {}".format(answer_dot_line))

    compilation = []      
    for entry in dict_json:
        compilation.append({"dataset_id": "figureqa-train1-v1", "image_id": dict_json[entry][1][13:], "graph_type": dict_json[entry][2][12:], "passage": dict_json[entry][3][9:], "question": dict_json[entry][4][10:], "answer": dict_json[entry][5][8:]})

    f = open("samples.json", 'r')
    samples = json.load(f)
    f.close()

    for sample in compilation:
        samples.append(sample)

    f = open("samples.json", 'w')
    json.dump(samples, f)
    f.close()


data_gen(1000)

print("Siddharth.py done")