from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import psycopg2
from psycopg2 import sql
import folium

# Initialize Flask application
app = Flask(__name__)

############# START POSTGRES CONNECTION ############
# Info for the connection with PostgreSQL in RDS
PGEND_POINT = 'database-1.cfwmkaw8o6bp.us-east-1.rds.amazonaws.com'  # End Point
PGDATABASE_NAME = 'heart_attack_prediction_db'  # Database name
PGUSER_NAME = 'postgres'
PGPASSWORD = 'B00TC4MP'


def connect():
    conn_string = f"host={PGEND_POINT} port=5432 dbname={PGDATABASE_NAME} user={PGUSER_NAME} password={PGPASSWORD}"
    conn = psycopg2.connect(conn_string)
    print("Connected!")
    cursor = conn.cursor()
    return conn, cursor


def close_connection(conn, cursor):
    conn.commit()
    cursor.close()
    conn.close()
    print("Connection closed.")


conn, cursor = connect()

# SQL SELECT statement to retrieve all columns (*) from the table named encodedtable2
query_hat_all = sql.SQL("""
SELECT * FROM encodedtable2;
""")
cur = conn.cursor()
cur.execute(query_hat_all)

# Put all the data in the encodedtable2 table into a DataFrame in pandas
encoded_df = pd.DataFrame(cur.fetchall(), columns=['Patient ID', 'Country', 'Capital', 'Age', 'Sex', 'Cholesterol',
                                                   'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
                                                   'Alcohol Consumption', 'Exercise Hours Per Week',
                                                   'Previous Heart Problems', 'Medication Use', 'Stress Level',
                                                   'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                                                   'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Continent',
                                                   'Hemisphere', 'Heart Attack Risk', 'Systolic Pressure',
                                                   'Diastolic Pressure', 'lat', 'long', 'Diet_Average', 'Diet_Healthy',
                                                   'Diet_Unhealthy'])

close_connection(conn, cursor)
############ END POSTGRES CONNECTION ############

# Separate features and target variable
X = encoded_df.drop(['Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 'Income',
                     'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Heart Attack Risk', 'Diet_Average',
                     'Diet_Healthy', 'Diet_Unhealthy', 'Country', 'Capital', 'lat', 'long', 'Continent',
                     'Patient ID', 'Hemisphere'], axis=1)
y = encoded_df['Heart Attack Risk']

# Create an instance of RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Resample the data
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Normalize the feature matrix X_resampled
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)

# Normalize the feature matrix after splitting
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the random forest classifier with default hyperparameters
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the entire training set with the best hyperparameters
best_rf_model = RandomForestClassifier(max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300,
                                        random_state=42)
best_rf_model.fit(X_train_scaled, y_train)

heart_trichol = {
    'Triglycerides': {'Abuja': 425.3950892857143, 'Bangkok': 429.88317757009344, 'Beijing': 413.32339449541286, 'Berlin': 416.52830188679246, 'Bogota': 405.5221445221445, 'Brasilia': 406.0, 'Buenos Aires': 419.7855626326964, 'Canberra': 419.87973273942094, 'Hanoi': 430.3270588235294, 'London': 419.6017505470459, 'Madrid': 409.66511627906976, 'New Delhi': 396.5194174757282, 'Ottawa': 406.05227272727274, 'Paris': 434.23766816143495, 'Pretoria': 417.2917647058824, 'Rome': 414.02088167053364, 'Seoul': 420.09290953545235, 'Tokyo': 426.0323325635104, 'Washington': 434.1666666666667, 'Wellington': 409.0206896551724},
    'Cholesterol': {'Abuja': 262.20535714285717, 'Bangkok': 262.1378504672897, 'Beijing': 258.0091743119266, 'Berlin': 258.0607966457023, 'Bogota': 261.84615384615387, 'Brasilia': 257.2705627705628, 'Buenos Aires': 261.52016985138005, 'Canberra': 258.9977728285078, 'Hanoi': 264.4188235294118, 'London': 259.8336980306346, 'Madrid': 258.8558139534884, 'New Delhi': 256.08009708737865, 'Ottawa': 256.8477272727273, 'Paris': 264.9484304932735, 'Pretoria': 253.22588235294117, 'Rome': 260.6658932714617, 'Seoul': 259.6454767726161, 'Tokyo': 260.96535796766744, 'Washington': 268.85238095238094, 'Wellington': 253.34022988505748},
    'lat': {'Abuja': 10.0, 'Bangkok': 15.0, 'Beijing': 35.0, 'Berlin': 51.0, 'Bogota': 4.0, 'Brasilia': -10.0, 'Buenos Aires': -34.0, 'Canberra': -27.0, 'Hanoi': 16.0, 'London': 54.0, 'Madrid': 40.0, 'New Delhi': 20.0, 'Ottawa': 60.0, 'Paris': 46.0, 'Pretoria': -29.0, 'Rome': 42.8333, 'Seoul': 37.0, 'Tokyo': 36.0, 'Washington': 38.0, 'Wellington': -41.0},
    'long': {'Abuja': 8.0, 'Bangkok': 100.0, 'Beijing': 105.0, 'Berlin': 9.0, 'Bogota': -72.0, 'Brasilia': -55.0, 'Buenos Aires': -64.0, 'Canberra': 133.0, 'Hanoi': 106.0, 'London': -2.0, 'Madrid': -4.0, 'New Delhi': 77.0, 'Ottawa': -95.0, 'Paris': 2.0, 'Pretoria': 24.0, 'Rome': 12.8333, 'Seoul': 127.5, 'Tokyo': 138.0, 'Washington': -97.0, 'Wellington': 174.0}
}


# Define the route for the home page
@app.route('/')
def home():
    return render_template('heart_attack.html')

@app.route('/visuals')
def index():
    # Get all countries
    all_countries = list(heart_trichol['lat'].keys())

    # Create a base map centered on the world
    world_map = folium.Map(location=[0, 0], zoom_start=2)

    # Create a feature group for top triglycerides layer
    triglycerides_layer = folium.FeatureGroup(name='Top 3 Triglycerides')

    for city, value in sorted(heart_trichol['Triglycerides'].items(), key=lambda x: x[1], reverse=True)[:3]:
        folium.Marker(location=[heart_trichol['lat'][city], heart_trichol['long'][city]],
                    popup=f'Triglycerides: {value}',
                    icon=folium.Icon(color='blue')).add_to(triglycerides_layer)

    # Create a feature group for top cholesterol layer
    cholesterol_layer = folium.FeatureGroup(name='Top 3 Cholesterol')

    for city, value in sorted(heart_trichol['Cholesterol'].items(), key=lambda x: x[1], reverse=True)[:3]:
        folium.Marker(location=[heart_trichol['lat'][city], heart_trichol['long'][city]],
                    popup=f'Cholesterol: {value}',
                    icon=folium.Icon(color='red')).add_to(cholesterol_layer)

    # Create a feature group for all countries layer
    all_countries_layer = folium.FeatureGroup(name='All Countries')

    for city in all_countries:
        folium.Marker(location=[heart_trichol['lat'][city], heart_trichol['long'][city]],
                    popup=f'Triglycerides: {heart_trichol["Triglycerides"].get(city, "N/A")}, Cholesterol: {heart_trichol["Cholesterol"].get(city, "N/A")}',
                    icon=folium.Icon(color='green')).add_to(all_countries_layer)

    # Add layers to the map
    triglycerides_layer.add_to(world_map)
    cholesterol_layer.add_to(world_map)
    all_countries_layer.add_to(world_map)

    # Add layer control
    folium.LayerControl().add_to(world_map)

    # Save the map to a temporary file
    map_file_path = 'templates/map.html'
    world_map.save(map_file_path)

    # Render the template with the map
    return render_template('layer_map.html', map_file_path=map_file_path)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cholesterol = int(request.form['cholesterol'])
        heart_rate = int(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        obesity = int(request.form['obesity'])
        alcohol_consumption = int(request.form['alcohol_consumption'])
        previous_heart_problems = int(request.form['previous_heart_problems'])
        medication_use = int(request.form['medication_use'])
        bmi = float(request.form['bmi'])
        triglycerides = int(request.form['triglycerides'])
        systolic_pressure = int(request.form['systolic_pressure'])
        diastolic_pressure = int(request.form['diastolic_pressure'])

        # Create a feature vector from the user input
        user_input = [[age, sex, cholesterol, heart_rate, diabetes, family_history, smoking, obesity,
                       alcohol_consumption, previous_heart_problems, medication_use, bmi, triglycerides,
                       systolic_pressure, diastolic_pressure]]

        # Scale the feature vector using the scaler fitted on the training data
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = best_rf_model.predict(user_input_scaled)

        # Determine the prediction message
        if prediction == 1:
            prediction_message = "High risk of heart attack"
        else:
            prediction_message = "Low risk of heart attack"

        # Get the test data accuracy
        test_accuracy = round(best_rf_model.score(X_test_scaled, y_test) * 100, 2)


        # Render the prediction result page with the prediction message and test accuracy
        return render_template('result.html', prediction_message=prediction_message, test_accuracy=test_accuracy)


if __name__ == '__main__':
    app.run(debug=True)
