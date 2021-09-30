import json
import numpy as np
from random import randrange, sample, choice

f = open("../train_metadata.json", 'r')
graphs = json.load(f)
f.close()

tables = {}
samples = []
percentages = [50, 60, 75, 80, 90, 100]
incidents = ["a crash", "an incident", "a disaster", "unforeseen circumstances", "an issue", "a catastrophe", "a mishap", "unknown reasons", "failures", "loss"]
move = ["moved", "transferred", "went", "was reallocated", "shifted", "was sent", "was pushed", "was driven", "was reassigned"]
for i, img in enumerate(graphs):
	for text in img["texts"]:
		if text["text_function"] == "value_heading":
			value = text["text"].lower()
			if value[-1] == 's':
				value = value[:-1]
	tables[img["image"]] = np.array(img["table"])
	if len(tables[img["image"]]) <= 2:
		legend = 1
		entities = sample(range(0,len(tables[img["image"]][0])),2)
		phrase = ''
	else:
		legend = randrange(1,len(tables[img["image"]]))
		entities = sample(range(1,len(tables[img["image"]][0])),2)
		phrase = tables[img["image"]][legend][0]+" in "
	percent = choice(percentages)
	if percent == 100:
		portion = "all"
	elif percent == 50:
		portion = "half"
	else:
		portion = str(percent)+"%"
	passage = "Due to "+choice(incidents)+", "+portion+" of the "+value+" for "+phrase+tables[img["image"]][0][entities[0]]+" "+choice(move)+" to "+tables[img["image"]][0][entities[1]]+"."
	
	if phrase != '':
		values = tables[img["image"]][legend][1:].astype(float)
		ents = tables[img["image"]][0][1:]
	else:
		values = tables[img["image"]][legend].astype(float)
		ents = tables[img["image"]][0]
	min_max = choice([0,1]) #0 for min and 1 for max
	q_type = choice([1,2,3,4]) #1 for max/min value, 2 for max/min entity, 3 for lesser/more than, and 4 for smaller/greater than by how much
	if min_max == 1 and max(values) != float(tables[img["image"]][legend][entities[1]]):
		answer = round(float(tables[img["image"]][legend][entities[1]]) + percent/100*float(tables[img["image"]][legend][entities[0]]), 2)
		if answer > max(values):
			if q_type == 1:
				question = "What is the maximum "+value
			elif q_type == 2:
				question = "Who has the maximum "+value
				answer = tables[img["image"]][0][entities[1]]
			else:
				for e, val in enumerate(values):
					if max(values) == val:
						old_max = ents[e]
						break
				less_more = choice([0,1]) #0 for lesser than and 1 for more than
				if q_type == 3:
					question = "Is the "+value+" for "+phrase+tables[img["image"]][0][entities[1]]
					if less_more == 0:
						question += " lesser "
						answer = "no"
					else:
						question += " more "
						answer = "yes"
					question += "than "+old_max+"?"
				else:
					question = "By how much is the "+value+" for "+phrase
					if less_more == 0:
						question += old_max+" smaller than "+tables[img["image"]][0][entities[1]]+"?"
					else:
						question += tables[img["image"]][0][entities[1]]+" greater than "+old_max+"?"
					answer = round(round(float(tables[img["image"]][legend][entities[1]]) + percent/100*float(tables[img["image"]][legend][entities[0]]), 2) - max(values), 2)
		else:
			q_type = 0
	elif min_max == 0 and min(values) != float(tables[img["image"]][legend][entities[0]]):
		answer = round((1 - percent/100)*float(tables[img["image"]][legend][entities[0]]), 2)
		if answer < min(values):
			if q_type == 1:
				question = "What is the minimum "+value
			elif q_type == 2:
				question = "Who has the minimum "+value
				answer = tables[img["image"]][0][entities[0]]
			else:
				for e, val in enumerate(values):
					if min(values) == val:
						old_min = ents[e]
						break
				less_more = choice([0,1]) #0 for lesser than and 1 for more than
				if q_type == 3:
					question = "Is the "+value+" for "+phrase+tables[img["image"]][0][entities[0]]
					if less_more == 0:
						question += " lesser "
						answer = "yes"
					else:
						question += " more "
						answer = "no"
					question += "than "+old_min+"?"
				else:
					question = "By how much is the "+value+" for "+phrase
					if less_more == 1:
						question += old_min+" greater than "+tables[img["image"]][0][entities[0]]+"?"
					else:
						question += tables[img["image"]][0][entities[0]]+" smaller than "+old_min+"?"
					answer = round(min(values) - round((1 - percent/100)*float(tables[img["image"]][legend][entities[0]]), 2), 2)
		else:
			q_type = 0
	else:
		q_type = 0
	if q_type == 1 or q_type == 2:
		if phrase != '':
			question += " for "+tables[img["image"]][legend][0]
		question += "?"
	elif q_type == 0:
		q_entity = choice([0,1])
		question = "What is the "+value+" for "+phrase+tables[img["image"]][0][entities[q_entity]]+"?"
		if q_entity == 0:
			answer = round((1 - percent/100)*float(tables[img["image"]][legend][entities[0]]), 2)
		else:
			answer = round(float(tables[img["image"]][legend][entities[1]]) + percent/100*float(tables[img["image"]][legend][entities[0]]), 2)
	samples.append({"dataset_id": "dvqa_train", "image_id": img["image"], "graph_type": "bar", "passage": passage, "question": question, "answer": answer})
	if i == 999:
		break

f = open("samples.json", 'w')
json.dump(samples, f)
f.close()

print("Shivansh.py done")