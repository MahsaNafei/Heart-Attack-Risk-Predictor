 CREATE TABLE HeartAttackCapitals (
     	country VARCHAR(255),
	 	capital VARCHAR(255),
		patient_id VARCHAR(255) PRIMARY KEY,
	    age INT,
        sex VARCHAR(255),
        cholesterol INT,
        heart_rate INT,
        diabetes INT,
        family_history INT,
        smoking INT,
        obesity INT,
        alcohol_consumption INT,
        exercise_hours_per_week FLOAT,
        diet VARCHAR(255), 
        previous_heart_problems INT,  
        medication_use INT,  
        stress_level INT,
        sedentary_hours_per_day FLOAT,
        income INT,
        bmi FLOAT,
        triglycerides INT,
        physical_activity_days_per_week INT,
        sleep_hours_per_day INT,
        continent VARCHAR(255),
        hemisphere VARCHAR(255),
        heart_attack_risk INT,
	    lat FLOAT, 
	    lon FLOAT,
	    systolic_preassure INT,
	    diastolic_preassure INT
    );
	

	DROP TABLE IF EXISTS HeartAttackCapitals;
	
	
SELECT * FROM public.heartattackcapitals
ORDER BY patient_id ASC LIMIT 100