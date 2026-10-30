SELECT
    DATE_TRUNC('year', c.date)::date AS ds,
    SUM(s.sales) AS y
FROM public.m5_sales_train AS s
JOIN public.m5_calendar AS c
    ON s.d = c.d
WHERE s.item_id = 'HOUSEHOLD_1_230'
  AND s.store_id = 'WI_1'
GROUP BY DATE_TRUNC('year', c.date)
ORDER BY ds ASC;
