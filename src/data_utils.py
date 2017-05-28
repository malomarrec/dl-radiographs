import argparse
import json

# Instantiate the parser
parser = argparse.ArgumentParser(description='Get data cleaning arguments')

parser.add_argument('data_path', type=str,
                    help='Relative source data path, including file name')

parser.add_argument('out_path', type=str,
                    help='Relative data output path, including file name')

#Parse
args = parser.parse_args()


#Clean data

data_path = args.data_path
out_path = args.out_path

print(data_path)
print(out_path)


if "." in data_path:
	if data_path.split(".")[1] != "json":
		raise ValueError("Invalid Extension")
else:
	data_path + ".json"


if "." in out_path:
	if out_path.split(".")[1] != "csv":
		raise ValueError("Invalid Extension")
else:
	out_path + ".csv"


data = []

for line in open(data_path, 'r'):
	data.append(json.loads(line))	

print data[1000]






