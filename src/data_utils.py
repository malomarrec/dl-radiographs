import argparse
import json
def extract_data(data_path):
# Instantiate the parser
	parser = argparse.ArgumentParser(description='Get data cleaning arguments')

	parser.add_argument('data_path', type=str,
	                    help='Relative source data path, including file name')

	# parser.add_argument('out_path', type=str,
	#                     help='Relative data output path, including file name')

	#Parse
	args = parser.parse_args()


	#Clean data

	data_path = args.data_path
	out_path = args.out_path

	print(data_path)
	print(out_path)


	if "." in data_path:
		print("gcjgfx", data_path.split(".")[-1])
		if data_path.split(".")[-1] != "json":
			raise ValueError("Invalid Extension")
	else:
		data_path + ".json"


	# if "." in out_path:
	# 	print("qwerty", out_path.split(".")[-1])
	# 	if out_path.split(".")[-1] != "csv" and out_path.split(".")[-1] != "txt":
	# 		raise ValueError("Invalid Extension")
	# else:
	# 	out_path + ".csv"


	data = []

	for line in open(data_path, 'r'):
		data.append(json.loads(line))	

	print len(data)
	return data






