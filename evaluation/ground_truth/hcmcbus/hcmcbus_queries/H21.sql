SELECT
    DATE_TRUNC('year', d.date)::date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '55'
  AND TRIM(d.operator) = 'XKSG'
  AND TRIM(d.vehicle_group) = '4_CNG.2'
GROUP BY DATE_TRUNC('year', d.date)
ORDER BY ds ASC;
