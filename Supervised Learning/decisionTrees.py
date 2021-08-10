# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
	total = 0
	for group in groups:
		total += len(group)

	n_instances = float(total)
	print()
	print(f"no of instances = {n_instances}")
	
	# sum weighted Gini index for each group
	gini = 0.0
	
	for group in groups:
		print(f"group = {group}")
		size = float(len(group))
		print(f"size = {size}")
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0

		# score the group based on the score for each class
		for class_val in classes:
			print(f"class_val = {class_val}")
			p = [row[-1] for row in group].count(class_val) / size
			print(f"p = {p}")
			score += p * p
			print(f"score = {score}")
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
		print(gini)
	return gini
 

# test Gini values
# myList = [[1,0], [2,5]]
# print(len(myList))
gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1])
gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1])

