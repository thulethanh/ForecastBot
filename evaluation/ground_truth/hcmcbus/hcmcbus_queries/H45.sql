SELECT
    d.date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '148'
  AND TRIM(d.vehicle_group) = '3_CNG'
GROUP BY d.date
ORDER BY ds ASC;
