import pandas
import numpy as np
import csv

def writeCsv(train_stance, train_bodies, validation_stance, validation_bodies):
	print("train stance length:")
	print(len(train_stance))
	print("train bodies length:")
	print(len(train_bodies))
	print("validation stance length")
	print(len(validation_stance))
	print("validation bodies length")
	print(len(validation_bodies))

	with open("fnc-1/split/train_stances.csv","w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Headline","Body ID","Stance"])
		writer.writerows(train_stance)
		csvfile.close()
	
	with open("fnc-1/split/validation_stances.csv","w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Headline","Body ID","Stance"])
		writer.writerows(validation_stance)
		csvfile.close()

	with open("fnc-1/split/train_bodies.csv","w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Body ID","articleBody"])
		writer.writerows(train_bodies)
		csvfile.close()

	with open("fnc-1/split/validation_bodies.csv","w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Body ID","articleBody"])
		writer.writerows(validation_bodies)
		csvfile.close()


def splitFile():
	train_stance_csv = pandas.read_csv("fnc-1/train_stances.csv")
	train_bodies_csv = pandas.read_csv("fnc-1/train_bodies.csv")

	stance_unrelated = []
	stance_agree = []
	stance_disagree = []
	stance_discuss = []

	bodies_unrelated = []
	bodies_agree = []
	bodies_disagree = []
	bodies_discuss = []

	for line in train_stance_csv.values:
		if(line[2] == "unrelated"):
			stance_unrelated.append(line)
			bodies_unrelated.append(train_bodies_csv[train_bodies_csv["Body ID"] == line[1]].values[0])
		if(line[2] == "agree"):
	 		stance_agree.append(line)
	 		bodies_agree.append(train_bodies_csv[train_bodies_csv["Body ID"] == line[1]].values[0])
		if(line[2] == "disagree"):
	 		stance_disagree.append(line)
	 		bodies_disagree.append(train_bodies_csv[train_bodies_csv["Body ID"] == line[1]].values[0])
		if(line[2] == "discuss"):
	 		stance_discuss.append(line)
	 		bodies_discuss.append(train_bodies_csv[train_bodies_csv["Body ID"] == line[1]].values[0])
	
	# print("train data size:")
	# print(train_stance_csv.size)
	
	stance_unrelated_len = int(len(stance_unrelated) * 0.9)
	stance_agree_len = int(len(stance_agree) * 0.9)
	stance_disagree_len = int(len(stance_disagree) * 0.9)
	stance_discuss_len = int(len(stance_discuss) * 0.9)

	bodies_unrelated_len = int(len(bodies_unrelated) * 0.9)
	bodies_agree_len = int(len(bodies_agree) * 0.9)
	bodies_disagree_len = int(len(bodies_disagree) * 0.9)
	bodies_discuss_len = int(len(bodies_discuss) * 0.9)
	
	totoal_train_stance = np.vstack((stance_unrelated[0 : stance_unrelated_len], stance_agree[0 : stance_agree_len], stance_disagree[0 : stance_disagree_len], stance_discuss[0 : stance_discuss_len]))

	totoal_train_bodies = np.vstack((bodies_unrelated[0 : bodies_unrelated_len], bodies_agree[0 : bodies_agree_len], bodies_disagree[0 : bodies_disagree_len], bodies_discuss[0 : bodies_discuss_len]))

	totoal_validation_stance = np.vstack((stance_unrelated[stance_unrelated_len:], stance_agree[stance_agree_len:], stance_disagree[stance_disagree_len:], stance_discuss[stance_discuss_len:]))

	totoal_validation_bodies = np.vstack((bodies_unrelated[bodies_unrelated_len:], bodies_agree[bodies_agree_len:], bodies_disagree[bodies_disagree_len:], bodies_discuss[bodies_discuss_len:]))
	
	writeCsv(totoal_train_stance, totoal_train_bodies, totoal_validation_stance, totoal_validation_bodies)


splitFile()