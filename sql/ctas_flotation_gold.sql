USE silica_dw;
GO

IF NOT EXISTS (SELECT 1 FROM sys.external_file_formats WHERE name='parquet_ff')
  CREATE EXTERNAL FILE FORMAT parquet_ff WITH (FORMAT_TYPE = PARQUET);
GO

IF OBJECT_ID('dbo.flotation_gold','U') IS NOT NULL DROP EXTERNAL TABLE dbo.flotation_gold;
GO

CREATE EXTERNAL TABLE dbo.flotation_gold
WITH (
  DATA_SOURCE = silica_lake,
  LOCATION    = 'gold/flotation_gold',
  FILE_FORMAT = parquet_ff
)
AS
SELECT
  TRY_CONVERT(datetime2(0), [date])                           AS date_ts,
  TRY_CAST([% Iron Feed]                  AS float)           AS iron_feed_pct,
  TRY_CAST([% Silica Feed]                AS float)           AS silica_feed_pct,
  TRY_CAST([Starch Flow]                  AS float)           AS starch_flow,
  TRY_CAST([Amina Flow]                   AS float)           AS amina_flow,
  TRY_CAST([Ore Pulp Flow]                AS float)           AS ore_pulp_flow,
  TRY_CAST([Ore Pulp pH]                  AS float)           AS ore_pulp_ph,
  TRY_CAST([Ore Pulp Density]             AS float)           AS ore_pulp_density,
  TRY_CAST([Flotation Column 01 Air Flow] AS float)           AS col01_air_flow,
  TRY_CAST([Flotation Column 02 Air Flow] AS float)           AS col02_air_flow,
  TRY_CAST([Flotation Column 03 Air Flow] AS float)           AS col03_air_flow,
  TRY_CAST([Flotation Column 04 Air Flow] AS float)           AS col04_air_flow,
  TRY_CAST([Flotation Column 05 Air Flow] AS float)           AS col05_air_flow,
  TRY_CAST([Flotation Column 06 Air Flow] AS float)           AS col06_air_flow,
  TRY_CAST([Flotation Column 07 Air Flow] AS float)           AS col07_air_flow,
  TRY_CAST([Flotation Column 01 Level]    AS float)           AS col01_level,
  TRY_CAST([Flotation Column 02 Level]    AS float)           AS col02_level,
  TRY_CAST([Flotation Column 03 Level]    AS float)           AS col03_level,
  TRY_CAST([Flotation Column 04 Level]    AS float)           AS col04_level,
  TRY_CAST([Flotation Column 05 Level]    AS float)           AS col05_level,
  TRY_CAST([Flotation Column 06 Level]    AS float)           AS col06_level,
  TRY_CAST([Flotation Column 07 Level]    AS float)           AS col07_level,
  TRY_CAST([% Iron Concentrate]           AS float)           AS iron_conc_pct,
  TRY_CAST([% Silica Concentrate]         AS float)           AS silica_conc_pct
FROM OPENROWSET(
  BULK 'bronze/raw/MiningProcess_Flotation_Plant_Database.csv',
  DATA_SOURCE = 'silica_lake',
  FORMAT = 'CSV',
  PARSER_VERSION = '2.0',
  HEADER_ROW = TRUE
)
WITH (
  [date]                          NVARCHAR(64),
  [% Iron Feed]                   NVARCHAR(64),
  [% Silica Feed]                 NVARCHAR(64),
  [Starch Flow]                   NVARCHAR(64),
  [Amina Flow]                    NVARCHAR(64),
  [Ore Pulp Flow]                 NVARCHAR(64),
  [Ore Pulp pH]                   NVARCHAR(64),
  [Ore Pulp Density]              NVARCHAR(64),
  [Flotation Column 01 Air Flow]  NVARCHAR(64),
  [Flotation Column 02 Air Flow]  NVARCHAR(64),
  [Flotation Column 03 Air Flow]  NVARCHAR(64),
  [Flotation Column 04 Air Flow]  NVARCHAR(64),
  [Flotation Column 05 Air Flow]  NVARCHAR(64),
  [Flotation Column 06 Air Flow]  NVARCHAR(64),
  [Flotation Column 07 Air Flow]  NVARCHAR(64),
  [Flotation Column 01 Level]     NVARCHAR(64),
  [Flotation Column 02 Level]     NVARCHAR(64),
  [Flotation Column 03 Level]     NVARCHAR(64),
  [Flotation Column 04 Level]     NVARCHAR(64),
  [Flotation Column 05 Level]     NVARCHAR(64),
  [Flotation Column 06 Level]     NVARCHAR(64),
  [Flotation Column 07 Level]     NVARCHAR(64),
  [% Iron Concentrate]            NVARCHAR(64),
  [% Silica Concentrate]          NVARCHAR(64)
) AS src;
GO

SELECT TOP 10 * FROM dbo.flotation_gold;
GO
