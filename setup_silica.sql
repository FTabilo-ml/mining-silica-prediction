IF DB_ID('silica_dw') IS NULL CREATE DATABASE silica_dw;
GO
USE silica_dw;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_scoped_credentials WHERE name='msi_cred')
  CREATE DATABASE SCOPED CREDENTIAL msi_cred WITH IDENTITY='Managed Identity';
GO

IF NOT EXISTS (SELECT 1 FROM sys.external_data_sources WHERE name='silica_lake')
  CREATE EXTERNAL DATA SOURCE silica_lake
  WITH ( TYPE = HADOOP,
         LOCATION = 'abfss://datalake@stsilicadataft.dfs.core.windows.net',
         CREDENTIAL = msi_cred );
GO

PRINT 'PARQUET sample';
SELECT TOP 10 *
FROM OPENROWSET(
  BULK 'silver/processed/**/*.parquet',
  DATA_SOURCE='silica_lake',
  FORMAT='PARQUET'
) AS p;
GO

PRINT 'CSV sample';
SELECT TOP 10 *
FROM OPENROWSET(
  BULK 'bronze/raw/MiningProcess_Flotation_Plant_Database.csv',
  DATA_SOURCE='silica_lake',
  FORMAT='CSV',
  PARSER_VERSION='2.0',
  HEADER_ROW=TRUE
)
WITH (
  [\% Iron Feed]        FLOAT,
  [\% Silica Feed]      FLOAT,
  [Starch Flow]         FLOAT,
  [Amina Flow]          FLOAT,
  [Ore Pulp pH]         FLOAT,
  [Air Flow]            FLOAT,
  [Ore Pulp Density]    FLOAT,
  [Silica Concentrate]  FLOAT
) AS t;
GO
