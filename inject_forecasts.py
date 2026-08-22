import json
import gzip
import base64
import os

def inject_forecasts():
    forecast_path = "quantconnect_backtest/egarch_forecasts.json"
    main_py_path = "quantconnect_backtest/main.py"
    
    if not os.path.exists(forecast_path):
        print(f"Error: {forecast_path} not found.")
        return
        
    with open(forecast_path, "r") as f:
        data = f.read()
        
    compressed = gzip.compress(data.encode('utf-8'))
    b64_str = base64.b64encode(compressed).decode('utf-8')
    
    with open(main_py_path, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith("compressed_data ="):
            lines[i] = f'compressed_data = "{b64_str}"\n'
            break
            
    with open(main_py_path, "w") as f:
        f.writelines(lines)
        
    print("Successfully injected forecasts into main.py")

if __name__ == "__main__":
    inject_forecasts()
