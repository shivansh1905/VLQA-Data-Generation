import simplejson
import re
import cv2
import numpy as np
from tqdm import tqdm

FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)
shapes = ['triangle', 'square', 'circle', 'item']
colors = {'black': [0, 0, 0], 'yellow': [0, 255, 255], 'blue': [255, 153, 0]}

f = open("train.json", 'r')
train = f.read()
f.close()
touching_data = []

def grabJSON(s):
    """Takes the largest bite of JSON from the string.
       Returns (object_parsed, remaining_string)
    """
    decoder = simplejson.JSONDecoder()
    obj, end = decoder.raw_decode(s)
    end = WHITESPACE.match(s, end).end()
    return obj, s[end:]

while True:
    obj, remaining = grabJSON(train)
    if obj['sentence'].lower().find('touching') != -1 and obj['sentence'].lower().find('there is a ') != -1 and obj['sentence'].lower().find('and') == -1 and obj['sentence'].lower().find('no') == -1 and obj['sentence'].lower().find('closely') == -1 and obj['sentence'].lower().find('box') == -1:
    	touching_data.append(obj)
    train = remaining
    if not remaining.strip():
        break

for sample in tqdm(touching_data):
	img = cv2.imread('images/'+sample['directory']+'/train-'+sample['identifier']+'-0.png', cv2.IMREAD_COLOR)
	mask = np.full(img.shape[:3], 211, dtype="uint8")
	cv2.rectangle(mask, (100, 0), (149, 99), (128,128,128), -1)
	cv2.rectangle(mask, (250, 0), (299, 99), (128,128,128), -1)

	if sample['sentence'].lower().find('wall') != -1 or sample['sentence'].lower().find('edge') != -1:
		if sample['sentence'].lower().split(' ')[3] in shapes:
			color = ''
			obj = sample['sentence'].lower().split(' ')[3]
		elif sample['sentence'].lower().split(' ')[3] in colors and sample['sentence'].lower().split(' ')[4] in shapes:
			color = sample['sentence'].lower().split(' ')[3]
			obj = sample['sentence'].lower().split(' ')[4]
		elif sample['sentence'].lower().split(' ')[3] not in colors and sample['sentence'].lower().split(' ')[4] in colors and sample['sentence'].lower().split(' ')[5] in shapes:
			color = sample['sentence'].lower().split(' ')[4]
			obj = sample['sentence'].lower().split(' ')[5]

		for i, box in enumerate(sample['structured_rep']):
			for shape in box:
				shape_color = shape['color'].lower()
				if shape_color == '#0099ff':
					shape_color = 'blue'
				if shape['type'] == obj and shape_color == color:
					if sample['label'] == 'true':
						if shape['y_loc'] == 0 or shape['y_loc'] + shape['size'] == 100 or shape['x_loc'] == 0 or shape['x_loc'] + shape['size'] == 100:
							if shape['type'] == 'square':
								cv2.rectangle(mask, (i*150+shape['x_loc'] + 3, shape['y_loc'] + 3), (i*150+shape['x_loc'] + shape['size'] - 4, shape['y_loc'] + shape['size'] - 4), (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
							elif shape['type'] == 'circle':
								cv2.circle(mask, (i*150+shape['x_loc'] + shape['size']//2, shape['y_loc'] + shape['size']//2), shape['size']//2 - 2, (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
					else:
						if shape['x_loc'] < 100 - (shape['x_loc'] + shape['size']):
							x = shape['x_loc']
							xa = 0
						else:
							x = 100 - (shape['x_loc'] + shape['size'])
							xa = 100 - (shape['size'] - 1)
						if shape['y_loc'] < 100 - (shape['y_loc'] + shape['size']):
							y = shape['y_loc']
							ya = 0
						else:
							y = 100 - (shape['y_loc'] + shape['size'])
							ya = 100 - (shape['size'] - 1)
						if x < y:
							x = xa
							y = shape['y_loc']
						else:
							y = ya
							x = shape['x_loc']
						if shape['type'] == 'square':
							cv2.rectangle(mask, (i*150+x, y), (i*150+x + shape['size'] - 1, y + shape['size'] - 1), (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
						elif shape['type'] == 'circle':
							cv2.circle(mask, (i*150+x + shape['size']//2, y + shape['size']//2), shape['size']//2, (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
						elif shape['type'] == 'triangle':
							points = np.array([[i*150+x, y + 0.866*(shape['size'] - 1)], [i*150+x + shape['size'] - 1, y + 0.866*(shape['size'] - 1)], [i*150+x + (shape['size'] - 1)//2, y]], np.int32)
							cv2.fillPoly(mask, [points], (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]))
				else:
					if shape['type'] == 'square':
						cv2.rectangle(mask, (i*150+shape['x_loc'], shape['y_loc']), (i*150+shape['x_loc'] + shape['size'] - 1, shape['y_loc'] + shape['size'] - 1), (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
					elif shape['type'] == 'circle':
						cv2.circle(mask, (i*150+shape['x_loc'] + shape['size']//2, shape['y_loc'] + shape['size']//2), shape['size']//2, (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]), -1)
					elif shape['type'] == 'triangle':
						points = np.array([[i*150+shape['x_loc'], shape['y_loc'] + 0.866*(shape['size'] - 1)], [i*150+shape['x_loc'] + shape['size'] - 1, shape['y_loc'] + 0.866*(shape['size'] - 1)], [i*150+shape['x_loc'] + (shape['size'] - 1)//2, shape['y_loc']]], np.int32)
						cv2.fillPoly(mask, [points], (colors[shape_color][0],colors[shape_color][1],colors[shape_color][2]))
		cv2.imwrite('manipulated_images/train-'+sample['directory']+'-'+sample['identifier']+'.png', mask)