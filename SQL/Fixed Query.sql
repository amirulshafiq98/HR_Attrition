-- This entire SQL block, including the 6-month filtering, generates your full_summary.csv
-- Concat tables into 1 for easy viewing (This still seems to be the initial source, but all transformations are below)
 Select
  s.*,
  l.absence_code,
  l.approval_status,
  l.leave_type,
  l.request_date
 from snapshots_updated as s left join leave_requests as l
 ON s.employee_id = l.employee_id;

-----------------------------------------------------------------------------------

-- Step 0: (Optional) Ensure the final table is dropped for a clean re-run if creating a new table
 DROP TABLE IF EXISTS employee_monthly_churn_data; -- (This was the old name, now it's full_summary.csv)

-- CTE 1: Count snapshots for each employee to identify those with sufficient history
WITH EmployeeSnapshotCounts AS (
    SELECT
        employee_id,
        COUNT(DISTINCT snapshot_date) AS num_snapshots,
        MIN(snapshot_date) AS first_snapshot_date,
        MAX(snapshot_date) AS last_snapshot_date,
        MAX(termination_date) AS employee_termination_date -- Get the actual termination date for the employee
    FROM
        snapshots_updated
    GROUP BY
        employee_id
    -- HAVING
        -- COUNT(DISTINCT snapshot_date) >= 6 -- Filter for employees with at least 6 monthly snapshots - THIS HAPPENS HERE
),

-- CTE 2: Select all monthly snapshots for the qualifying employees
QualifyingSnapshots AS (
    SELECT
        s.*
    FROM
        snapshots_updated AS s
    JOIN
        EmployeeSnapshotCounts AS esc ON s.employee_id = esc.employee_id
),

-- CTE 3: Re-calculate leave summary for each snapshot, ensuring it only considers requests up to that snapshot date
-- This means joining `leave_requests` to `snapshots_updated` directly
EmployeeLeaveSummaryPerSnapshot AS (
    SELECT
        qs.employee_id,
        qs.snapshot_date,
        SUM(CASE WHEN lr.leave_type = 'Vacation' AND lr.request_date <= qs.snapshot_date THEN 1 ELSE 0 END) AS vacation_leave_upto_snapshot,
        SUM(CASE WHEN lr.leave_type = 'Parental' AND lr.request_date <= qs.snapshot_date THEN 1 ELSE 0 END) AS parental_leave_upto_snapshot,
        SUM(CASE WHEN lr.leave_type = 'Sick' AND lr.request_date <= qs.snapshot_date THEN 1 ELSE 0 END) AS sick_leave_upto_snapshot,
        SUM(CASE WHEN lr.leave_type = 'Personal' AND lr.request_date <= qs.snapshot_date THEN 1 ELSE 0 END) AS personal_leave_upto_snapshot
    FROM
        QualifyingSnapshots AS qs
    LEFT JOIN
        leave_requests AS lr ON qs.employee_id = lr.employee_id
    GROUP BY
        qs.employee_id,
        qs.snapshot_date
)

-- Final SELECT: Combine all features and define the target variable for each monthly snapshot
-- The target_variable for a given snapshot_date means: Did this employee churn within the next 6 months from this snapshot?
-- This is the query that will output your full_summary.csv
SELECT
    qs.employee_id,
    qs.snapshot_date, -- Keep all monthly snapshot dates
    qs.age,
    qs.department,
    qs.business_unit,
    qs.job_title,
    qs.location,
    qs.base_salary,
    qs.bonus_eligible,
    qs.bonus_pct,
    qs.equity_grant,
    qs.equity_pct,
    qs.employment_type,
    qs.hire_date,
    qs.ethnicity,
    qs.marital_status,
    qs.education_level,
    qs.pay_frequency,
    qs.veteran_status,
    qs.disability_status,
    qs.cost_center,
    qs.fte,
    qs.exemption_status,
    qs.high_potential_flag,
    qs.succession_plan_status,
    qs.aihr_certified,
    qs.promotion_count,
    qs.tenure_months, -- Tenure at this specific snapshot_date
    qs.performance_rating,
    qs.engagement_score,
    qs.risk_of_exit_score,
    qs.current_salary,
    qs.training_count,
    qs.job_level,
    qs.last_training_date, -- Keep for Python calculation if desired, or calculate here

    -- Leave summaries up to this specific snapshot date
    ISNULL(els.vacation_leave_upto_snapshot, 0) AS vacation_leave,
    ISNULL(els.sick_leave_upto_snapshot, 0) AS sick_leave,
    ISNULL(els.personal_leave_upto_snapshot, 0) AS personal_leave,
    ISNULL(els.parental_leave_upto_snapshot, 0) AS parental_leave,

    -- Calculate months_since_last_training relative to this snapshot_date
    CASE
        WHEN qs.last_training_date IS NULL THEN 99999
        WHEN DATEDIFF(month, qs.last_training_date, qs.snapshot_date) < 0 THEN 0
        ELSE DATEDIFF(month, qs.last_training_date, qs.snapshot_date)
    END AS months_since_last_training,

    -- Define the target variable for each monthly snapshot:
    -- 1 if employee terminates within the next 6 months from this snapshot_date, otherwise 0
    CASE
        WHEN esc.employee_termination_date IS NOT NULL AND
             esc.employee_termination_date > qs.snapshot_date AND -- Ensure termination is after the snapshot
             esc.employee_termination_date <= DATEADD(month, 6, qs.snapshot_date) -- Ensure termination is within next 6 months
        THEN 1
        ELSE 0
    END AS target_variable,
    
    -- *** ADD THIS NEW COLUMN HERE: ever_terminated_flag ***
    CASE
        WHEN esc.employee_termination_date IS NOT NULL THEN 1 -- If employee_termination_date exists, they terminated
        ELSE 0                                                 -- Otherwise, they have not terminated (based on available data)
    END AS ever_terminated_flag, -- This exact name to match your Python code!
    
    esc.employee_termination_date AS termination_date -- Include the actual termination date for reference
    -- DATEDIFF(day, qs.snapshot_date, esc.employee_termination_date) AS days_until_departure -- You can calculate this in Python if needed

-- (Optional) If you want to create a new table in your database for this dataset:
-- INTO employee_monthly_churn_prediction_data -- (This would be if it were an intermediate table, but it's full_summary.csv)

FROM
    QualifyingSnapshots AS qs
JOIN
    EmployeeSnapshotCounts AS esc ON qs.employee_id = esc.employee_id
LEFT JOIN
    EmployeeLeaveSummaryPerSnapshot AS els ON qs.employee_id = els.employee_id AND qs.snapshot_date = els.snapshot_date
WHERE
    NOT (esc.num_snapshots < 6 AND esc.employee_termination_date IS NOT NULL)
ORDER BY
    qs.employee_id, qs.snapshot_date;