SELECT
    DATE_TRUNC('quarter', d.date)::date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '122'
  AND TRIM(d.operator) = 'HTX-195'
GROUP BY DATE_TRUNC('quarter', d.date)
ORDER BY ds ASC;
