USE silica_dw;
GO

IF NOT EXISTS (SELECT 1 FROM sys.external_file_formats WHERE name='parquet_ff')
  CREATE EXTERNAL FILE FORMAT parquet_ff WITH (FORMAT_TYPE = PARQUET);
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
SELECT
  [date],
  [% Iron Feed],
  [% Silica Feed],
  [Starch Flow],
  [Amina Flow],
  [Ore Pulp Flow],
  [Ore Pulp pH],
  [Ore Pulp Density],
  [Flotation Column 01 Air Flow],
  [Flotation Column 02 Air Flow],
  [Flotation Column 03 Air Flow],
  [Flotation Column 04 Air Flow],
  [Flotation Column 05 Air Flow],
  [Flotation Column 06 Air Flow],
  [Flotation Column 07 Air Flow],
  [Flotation Column 01 Level],
  [Flotation Column 02 Level],
  [Flotation Column 03 Level],
  [Flotation Column 04 Level],
  [Flotation Column 05 Level],
  [Flotation Column 06 Level],
  [Flotation Column 07 Level],
  [% Iron Concentrate],
  [% Silica Concentrate]
FROM OPENROWSET(
  BULK 'silver/processed/*.parquet',
  DATA_SOURCE = 'silica_lake',
  FORMAT = 'PARQUET'
) AS p;
GO

SELECT TOP 10 * FROM dbo.features_all;
GO
