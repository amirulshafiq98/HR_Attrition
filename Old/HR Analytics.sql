-- Concat tables into 1 for easy viewing
Select 
	s.*,
	l.absence_code,
	l.approval_status,
	l.leave_type,
	l.request_date
from snapshots_updated as s left join leave_requests as l
ON s.employee_id = l.employee_id;


-------------------------------------------------------------------------------------

-- Here to ensure smooth re-runs
DROP TABLE IF EXISTS employee_full_summary;

-- Batch 1
WITH Employee_Leave_Summary AS (
    SELECT
        employee_id,
        SUM(CASE WHEN leave_type = 'Vacation' THEN 1 ELSE 0 END) AS vacation_leave,
        SUM(CASE WHEN leave_type = 'Parental' THEN 1 ELSE 0 END) AS parental_leave,
        SUM(CASE WHEN leave_type = 'Sick' THEN 1 ELSE 0 END) AS sick_leave,
        SUM(CASE WHEN leave_type = 'Personal' THEN 1 ELSE 0 END) AS personal_leave
    FROM
        leave_requests
    GROUP BY
        employee_id
),
-- SELECT * FROM Employee_Leave_Summary;

-- Add a GO statement to separate batches if running together
-- GO

-- Batch 2
Employee_Promotion_Date AS (
    SELECT
        employee_id,
        hire_date,
        last_promotion_date,
        DATEDIFF(month, hire_date, last_promotion_date) AS months_to_promote,
		CASE WHEN termination_date IS NOT NULL THEN 1 ELSE 0 END AS terminated_flag
    FROM
        snapshots_updated
    WHERE
        last_promotion_date IS NOT NULL 
		AND hire_date IS NOT NULL 
)
-- SELECT * FROM Employee_Promotion_Date ORDER BY employee_id ASC;

SELECT
    s.*,
    ISNULL(els.vacation_leave, 0) AS vacation_leave,
    ISNULL(els.sick_leave, 0) AS sick_leave,
    ISNULL(els.personal_leave, 0) AS personal_leave,
    ISNULL(els.parental_leave, 0) AS parental_leave,
    ISNULL(epd.months_to_promote, 0) AS months_to_promote,
    ISNULL(epd.terminated_flag, 0) AS terminated_flag,
	CASE
        WHEN s.last_training_date IS NULL THEN 99999 -- Still assign 99999 for truly NULL dates
        WHEN DATEDIFF(month, s.last_training_date, GETDATE()) < 0 THEN 0 -- If date is in future, treat as 0 months since
        ELSE DATEDIFF(month, s.last_training_date, GETDATE()) -- Otherwise, calculate normally
    END AS months_since_last_training
INTO
    employee_full_summary
FROM
    snapshots_updated as s
LEFT JOIN
    Employee_Leave_Summary as els
    ON s.employee_id = els.employee_id
LEFT JOIN
    Employee_Promotion_Date as epd
    ON s.employee_id = epd.employee_id;

-- Check columns with many NULL values
--SELECT
--    COUNT(*) AS TotalRows,
--    SUM(CASE WHEN manager_id IS NULL THEN 1 ELSE 0 END) AS Null_manager_id_Count,
--    CAST(SUM(CASE WHEN manager_id IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS Null_manager_id_Percent,
--    SUM(CASE WHEN termination_date IS NULL THEN 1 ELSE 0 END) AS Null_termination_date_Count,
--    CAST(SUM(CASE WHEN termination_date IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS Null_termination_date_Percent,
--    SUM(CASE WHEN last_promotion_date IS NULL THEN 1 ELSE 0 END) AS Null_last_promotion_date_Count,
--    CAST(SUM(CASE WHEN last_promotion_date IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS Null_last_promotion_date_Percent,
--    SUM(CASE WHEN last_training_date IS NULL THEN 1 ELSE 0 END) AS Null_last_training_date_Count,
--    CAST(SUM(CASE WHEN last_training_date IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS DECIMAL(5,2)) AS Null_last_training_date_Percent
--FROM
--    snapshots_updated;


ALTER TABLE employee_full_summary
DROP COLUMN manager_id, termination_date, last_promotion_date, last_training_date


SELECT * FROM employee_full_summary;

