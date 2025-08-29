USE silica_dw;
GO

IF OBJECT_ID('dbo.kpi_daily','V') IS NOT NULL DROP VIEW dbo.kpi_daily;
GO

CREATE VIEW dbo.kpi_daily AS
WITH src AS (
  SELECT
    TRY_CONVERT(date, [date])                                          AS day,
    TRY_CAST([Ore Pulp pH]            AS float)                        AS ore_pulp_ph,
    TRY_CAST([Starch Flow]            AS float)                        AS starch_flow,
    TRY_CAST([Amina Flow]             AS float)                        AS amina_flow,
    TRY_CAST([% Silica Concentrate]   AS float)                        AS silica_conc
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
  ) AS t
)
SELECT
  day,
  AVG(ore_pulp_ph)     AS pH_avg,
  AVG(starch_flow)     AS starch_flow_avg,
  AVG(amina_flow)      AS amina_flow_avg,
  AVG(silica_conc)     AS silica_conc_avg,
  COUNT(*)             AS rows_cnt
FROM src
WHERE day IS NOT NULL
GROUP BY day;
GO

SELECT TOP 7 * FROM dbo.kpi_daily ORDER BY day DESC;
GO
