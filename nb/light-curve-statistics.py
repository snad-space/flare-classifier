import pyarrow.parquet as pq


total_num_values = 0
total_num_rows = 0
total_uniques = 0
for path in sorted(Path('/home/lavrukhina/fields_data').glob('*.parquet')):
    with path.open('rb') as fh:
        parquet_file = pq.ParquetFile(fh)
        
        unique_oids = parquet_file.read(['oid']).column('oid').unique()
        total_uniques += len(unique_oids)
        
        metadata = parquet_file.metadata
        total_num_rows += metadata.num_rows
        
        num_row_groups = metadata.num_row_groups
        arrow_schema = metadata.schema.to_arrow_schema()
        mjd_field_idx = arrow_schema.names.index('mjd')
        
        for row_group_idx in range(num_row_groups):
            row_group = metadata.row_group(row_group_idx)
            column = row_group.column(mjd_field_idx)
            total_num_values += column.statistics.num_values

print(total_uniques, total_num_rows, total_num_values, total_num_values / total_num_rows)
# (54800806, 93660131, 4145625951, 44.2624402372446)
