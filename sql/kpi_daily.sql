USE silica_dw;
GO
IF OBJECT_ID('dbo.kpi_daily','V') IS NOT NULL DROP VIEW dbo.kpi_daily;
GO
CREATE VIEW dbo.kpi_daily AS
SELECT
  CAST(COALESCE(p.[timestamp], p.[event_time], p.[time], p.[date]) AS date) AS day,
  AVG(TRY_CAST(p.[Ore Pulp pH] AS float))          AS pH_avg,
  AVG(TRY_CAST(p.[Starch Flow] AS float))          AS starch_flow_avg,
  AVG(TRY_CAST(p.[Amina Flow] AS float))           AS amina_flow_avg,
  AVG(TRY_CAST(p.[% Silica Concentrate] AS float)) AS silica_conc_avg,
  COUNT(*)                                         AS rows_cnt
FROM OPENROWSET(
  BULK 'silver/processed/*.parquet',
  DATA_SOURCE='silica_lake',
  FORMAT='PARQUET'
) AS p
GROUP BY CAST(COALESCE(p.[timestamp], p.[event_time], p.[time], p.[date]) AS date);
GO

SELECT TOP 7 * FROM dbo.kpi_daily ORDER BY day DESC;
GO
