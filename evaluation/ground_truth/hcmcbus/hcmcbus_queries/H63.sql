SELECT
    DATE_TRUNC('year', d.date)::date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '99'
  AND TRIM(d.operator) = 'HTX-ThanhSon'
  AND TRIM(d.vehicle_group) = '2B.1_ML'
GROUP BY DATE_TRUNC('year', d.date)
ORDER BY ds ASC;
