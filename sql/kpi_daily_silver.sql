USE silica_dw;
GO

IF OBJECT_ID('dbo.kpi_daily','V') IS NOT NULL DROP VIEW dbo.kpi_daily;
GO

CREATE VIEW dbo.kpi_daily AS
SELECT
  TRY_CONVERT(date, [date])                               AS day,
  AVG(TRY_CAST([Ore Pulp pH]            AS float))        AS pH_avg,
  AVG(TRY_CAST([Starch Flow]            AS float))        AS starch_flow_avg,
  AVG(TRY_CAST([Amina Flow]             AS float))        AS amina_flow_avg,
  AVG(TRY_CAST([% Silica Concentrate]   AS float))        AS silica_conc_avg,
  COUNT(*)                                               AS rows_cnt
FROM OPENROWSET(
  BULK 'silver/processed/*.parquet',       -- sin ** (Synapse no admite comodines consecutivos)
  DATA_SOURCE = 'silica_lake',
  FORMAT = 'PARQUET'
) AS p
GROUP BY TRY_CONVERT(date, [date]);
GO

SELECT TOP 7 * FROM dbo.kpi_daily ORDER BY day DESC;
GO
