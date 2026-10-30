SELECT
    c.date AS ds,
    SUM(s.sales) AS y
FROM public.m5_sales_train AS s
JOIN public.m5_calendar AS c
    ON s.d = c.d
WHERE s.item_id = 'FOODS_3_540'
  AND s.store_id = 'CA_4'
GROUP BY c.date
ORDER BY ds ASC;
