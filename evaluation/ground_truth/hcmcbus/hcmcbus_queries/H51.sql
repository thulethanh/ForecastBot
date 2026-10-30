SELECT
    DATE_TRUNC('month', d.date)::date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '70'
GROUP BY DATE_TRUNC('month', d.date)
ORDER BY ds ASC;
