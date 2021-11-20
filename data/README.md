# anomaly-detection-benchmark dataset
Dataset requirements for using with benchmark:
1. Dataset Types: univariate and multivariate numerical data
2. File Type: CSV
3. Each dataset must have last column with heading **is_anomaly** and contains 0 as normal instance and 1 as anomaly.
4. Each CSV file must have first column with heading **timestamp**. If it is timeseries data then you already have timestamp column, otherwise add timestamp column with dummy values (these values will be ignored).
