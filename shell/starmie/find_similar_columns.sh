#!/bin/bash

# --- Configuration ---
MODEL_PATH="checkpoints/starmie/santos/model_drop_col_tfidf_entity_column_0.pt" 
DATA_DIR="data/santos/datalake"
TARGET_COLUMN="ticket_created_date_time"
# TARGET_TABLE="optional_specific_table.csv" # Optional: if we want to specify the table of the target column
TOP_N=30

echo "Finding columns similar to '$TARGET_COLUMN' in '$DATA_DIR' using model '$MODEL_PATH'"

python scripts/starmie/find_similar_columns.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --target_column_name "$TARGET_COLUMN" \
  --top_n "$TOP_N"
  # --target_table_name "$TARGET_TABLE"

# python scripts/starmie/find_similar_columns_enhanced.py \
#   --model_path "$MODEL_PATH" \
#   --data_dir "$DATA_DIR" \
#   --target_column_name "$TARGET_COLUMN" \
#   --top_n "$TOP_N"
#   # --target_table_name "$TARGET_TABLE"


echo "Script execution finished."