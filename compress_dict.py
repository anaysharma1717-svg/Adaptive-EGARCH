import json, gzip, base64, ast, re

with open('quantconnect_backtest/main.py', 'r', encoding='utf-8') as f:
    main_code = f.read()

# Find the dictionary
match = re.search(r'EGARCH_H21_FORECASTS = (\{.*?\})', main_code, re.DOTALL)
if match:
    dict_str = match.group(1)
    d = ast.literal_eval(dict_str)
    
    json_str = json.dumps(d)
    compressed = gzip.compress(json_str.encode('utf-8'))
    b64_str = base64.b64encode(compressed).decode('utf-8')
    
    print('Original length:', len(json_str))
    print('Base64 length:', len(b64_str))
    
    replacement = f"""import json, gzip, base64
compressed_data = "{b64_str}"
EGARCH_H21_FORECASTS = json.loads(gzip.decompress(base64.b64decode(compressed_data)).decode('utf-8'))"""
    
    new_main = main_code.replace(match.group(0), replacement)
    
    with open('quantconnect_backtest/main.py', 'w', encoding='utf-8') as f:
        f.write(new_main)
    print("Successfully compressed and injected into main.py")
else:
    print("Could not find dictionary")
