# Disaster Relief Optimizer

This project is a Disaster Relief Optimizer that predicts resource needs and provides a relief plan based on various disaster parameters.

## Project Structure

- `app.py`: The main Flask application file that handles web requests and renders the UI.
- `disasterrelief.py`: Contains the core logic for disaster prediction and relief optimization.
- `Natural_Disasters_in_India .csv`: Dataset used for training the disaster prediction model.
- `category_label_encoder.pkl`: Pickled LabelEncoder for disaster categories.
- `disaster_model.pkl`: Pickled machine learning model for disaster prediction.
- `static/`: Contains static files like CSS and JavaScript.
    - `static/style.css`: Stylesheet for the web application.
    - `static/script.js`: JavaScript for interactive elements and chart rendering.
- `templates/`: Contains HTML templates.
    - `templates/index.html`: The main HTML file for the application.
- Image files (e.g., `average_severity_per_year.png`, `disaster_categories_by_month.png`, etc.): Various visualizations generated from the disaster data.

## How to Run

1.  **Install Dependencies**: Make sure you have Python and pip installed. Then install the required Python packages:
    ```bash
    pip install Flask pandas scikit-learn matplotlib seaborn
    ```
2.  **Run the Application**: Navigate to the project directory and run the Flask application:
    ```bash
    python app.py
    ```
3.  **Access the Application**: Open your web browser and go to `http://127.0.0.1:5000/` (or the address shown in your terminal).

## Features

-   **Disaster Prediction**: Predicts severity scores based on disaster title, duration, location, and available inventory.
-   **Resource Optimization**: Provides a distribution plan and recommended additional units needed.
-   **Interactive UI**: User-friendly interface for inputting disaster details and viewing results.
-   **Data Visualization**: Displays various charts and graphs related to disaster data.
