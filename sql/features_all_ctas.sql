USE silica_dw;
GO

IF NOT EXISTS (SELECT 1 FROM sys.external_file_formats WHERE name='parquet_ff')
  CREATE EXTERNAL FILE FORMAT parquet_ff WITH (FORMAT_TYPE=PARQUET);
GO

IF OBJECT_ID('dbo.features_all','U') IS NOT NULL DROP EXTERNAL TABLE dbo.features_all;
GO

CREATE EXTERNAL TABLE dbo.features_all
WITH (
  DATA_SOURCE = silica_lake,
  LOCATION    = 'gold/features_all',
  FILE_FORMAT = parquet_ff
)
AS
SELECT *
FROM OPENROWSET(
  BULK 'silver/processed/features_parts/*.parquet',
  DATA_SOURCE='silica_lake',
  FORMAT='PARQUET'
) AS p;
GO

SELECT TOP 10 * FROM dbo.features_all;
GO
