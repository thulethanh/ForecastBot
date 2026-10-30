SELECT
    c.date AS ds,
    SUM(s.sales) AS y
FROM public.m5_sales_train AS s
JOIN public.m5_calendar AS c
    ON s.d = c.d
WHERE s.item_id = 'FOODS_3_629'
  AND s.store_id = 'TX_3'
  AND s.state_id = 'TX'
GROUP BY c.date
ORDER BY ds ASC;
