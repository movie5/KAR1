import json
with open("OBJ05166_PS3_K3A_NIA0338.json", 'r') as f:
    data = json.load(f)

num = data['features'][1]["properties"]["object_imcoords"]
parsed_numbers = [round(float(num), 2) for num in num.split(',')]
print(parsed_numbers)
