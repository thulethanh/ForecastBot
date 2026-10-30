SELECT
    DATE_TRUNC('quarter', c.date)::date AS ds,
    SUM(s.sales) AS y
FROM public.m5_sales_train AS s
JOIN public.m5_calendar AS c
    ON s.d = c.d
WHERE s.item_id = 'HOUSEHOLD_1_143'
  AND s.state_id = 'CA'
GROUP BY DATE_TRUNC('quarter', c.date)
ORDER BY ds ASC;
