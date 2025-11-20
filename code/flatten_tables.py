import pandas as pd
import os
import re
import warnings

# Suppress specific warnings
warnings.simplefilter("ignore")

INPUT_DIR = "processed_tables_clean"
OUTPUT_DIR = "output"

# Regex to capture years (e.g., 2018, 2018-2019, 2018–2019)
YEAR_PATTERN = r"(\d{4}(?:[-–]\d{4})?)"

def get_files():
    if not os.path.exists(INPUT_DIR):
        print(f"CRITICAL: Input directory '{INPUT_DIR}' does not exist.")
        return []
    return [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

def extract_year_and_metric(col_name):
    """
    Parses a column header to separate the Year from the Metric.
    """
    col_str = str(col_name)
    match = re.search(YEAR_PATTERN, col_str)
    
    if match:
        year = match.group(1).replace('–', '-') # Normalize en-dash to hyphen
        # Remove the year from the string to find the metric
        metric = col_str.replace(match.group(0), '').strip()
        
        # Clean up metric punctuation
        metric = metric.strip('()[]')
        
        if not metric:
            metric = "Value"
            
        return year, metric
    return None, None

def process_table(filename):
    file_path = os.path.join(INPUT_DIR, filename)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    if df.empty:
        return

    # 1. Identify Time Series vs Anchor Columns
    # We do NOT assume index 0 is the only anchor. 
    # Any column that yields a valid Year is 'data'. Everything else is 'id'.
    time_series_cols = []
    mappings = {} 
    
    for col in df.columns:
        year, metric = extract_year_and_metric(col)
        if year:
            time_series_cols.append(col)
            mappings[col] = (year, metric)
    
    # Define Anchors (id_vars) as anything that ISN'T a time series column
    id_vars = [c for c in df.columns if c not in time_series_cols]

    # 2. Execute Flattening Strategy
    if not time_series_cols:
        print(f"  [Pass-Through] No time-series detected in: {filename}")
        df_out = df.copy()
        
    else:
        print(f"  [Flattening] Time-series detected in: {filename} (Anchors: {id_vars})")
        
        # Melt
        df_melt = df.melt(id_vars=id_vars, value_vars=time_series_cols, var_name='original_header', value_name='temp_value')
        
        # Map Year/Metric
        metric_data = df_melt['original_header'].map(mappings)
        df_melt['Year'] = metric_data.apply(lambda x: x[0])
        df_melt['Metric'] = metric_data.apply(lambda x: x[1])
        
        # Pivot
        # We pivot on ALL anchors + Year. This preserves 'Sex', 'Age group', etc.
        pivot_index = id_vars + ['Year']
        
        try:
            df_out = df_melt.pivot_table(
                index=pivot_index, 
                columns='Metric', 
                values='temp_value', 
                aggfunc='first' 
            ).reset_index()
            df_out.columns.name = None
        except Exception as e:
            print(f"    ! Pivot Failed for {filename}: {e}. Saving long format.")
            df_out = df_melt.drop(columns=['original_header'])

    # 3. Smart Data Type Cleaning
    # Do NOT blindly convert everything to numeric. 
    # '95% CI' contains dashes (100-200) and will become NaN if coerced.
    for col in df_out.columns:
        if col in id_vars or col == 'Year':
            continue
            
        # check samples for non-numeric chars that aren't simple separators
        sample_values = df_out[col].dropna().astype(str)
        if sample_values.empty:
            continue
            
        # Heuristic: If column contains dashes inside the string (e.g. '111-222'), it's a range (String).
        # If it matches standard float/int regex, convert it.
        is_range = sample_values.str.contains(r'\d+[-–]\d+', regex=True).any()
        
        if not is_range:
            # Safe to coerce to number, turning 'F', 'x' into NaN
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = f"flat_{filename}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    df_out.to_csv(out_path, index=False)
    print(f"    -> Saved: {out_path}")

if __name__ == "__main__":
    files = get_files()
    print(f"Processing {len(files)} files...")
    for f in files:
        process_table(f)