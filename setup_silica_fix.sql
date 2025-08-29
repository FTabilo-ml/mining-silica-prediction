-- Crea DB si no existe y úsala
IF DB_ID('silica_dw') IS NULL
  CREATE DATABASE silica_dw;
GO
USE silica_dw;
GO

-- 1) MASTER KEY (requerida para credenciales)
--   Elige una contraseña segura que puedas guardar.
IF NOT EXISTS (SELECT * FROM sys.symmetric_keys WHERE name = '##MS_DatabaseMasterKey##')
  CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'SilicaDbMK_2025!Cambiar';
GO

-- 2) CREDENCIAL MSI (usa la Managed Identity del workspace)
IF NOT EXISTS (SELECT 1 FROM sys.database_scoped_credentials WHERE name='msi_cred')
  CREATE DATABASE SCOPED CREDENTIAL msi_cred WITH IDENTITY='Managed Identity';
GO

-- 3) EXTERNAL DATA SOURCE al Lake (ADLS Gen2)
IF NOT EXISTS (SELECT 1 FROM sys.external_data_sources WHERE name='silica_lake')
  CREATE EXTERNAL DATA SOURCE silica_lake
  WITH (
    TYPE = HADOOP,
    LOCATION = 'abfss://datalake@stsilicadataft.dfs.core.windows.net',
    CREDENTIAL = msi_cred
  );
GO

-- 4) PRUEBA PARQUET (silver)
PRINT 'PARQUET sample';
SELECT TOP 10 *
FROM OPENROWSET(
  BULK 'silver/processed/**/*.parquet',
  DATA_SOURCE='silica_lake',
  FORMAT='PARQUET'
) AS p;
GO

-- 5) PRUEBA CSV (bronze)
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
