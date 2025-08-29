USE silica_dw;
GO

-- MASTER KEY (si no existe)
IF NOT EXISTS (SELECT * FROM sys.symmetric_keys WHERE name = '##MS_DatabaseMasterKey##')
  CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'SilicaDbMK_2025!Cambiar';
GO

-- CREDENCIAL MSI (si no existe)
IF NOT EXISTS (SELECT 1 FROM sys.database_scoped_credentials WHERE name='msi_cred')
  CREATE DATABASE SCOPED CREDENTIAL msi_cred WITH IDENTITY='Managed Identity';
GO

-- Limpia data source previo si quedó a medio crear
IF EXISTS (SELECT 1 FROM sys.external_data_sources WHERE name='silica_lake')
  DROP EXTERNAL DATA SOURCE silica_lake;
GO

-- *** CREA EL DATA SOURCE SIN TYPE=HADOOP ***
CREATE EXTERNAL DATA SOURCE silica_lake
WITH (
  LOCATION   = 'abfss://datalake@stsilicadataft.dfs.core.windows.net',
  CREDENTIAL = msi_cred
);
GO

-- PRUEBA PARQUET
PRINT 'PARQUET sample';
SELECT TOP 10 *
FROM OPENROWSET(
  BULK 'silver/processed/**/*.parquet',
  DATA_SOURCE='silica_lake',
  FORMAT='PARQUET'
) AS p;
GO

-- PRUEBA CSV
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
