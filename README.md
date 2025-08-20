# NBA_Analytics_Dashboard

## Description
This project is an interactive dashboard designed to analyze and predict NBA player performances for the 2025-26 season. Leveraging historical data retrieved via the NBA API, a machine learning model (Random Forest Regressor) forecasts statistics such as points, rebounds, assists, and field goal percentage. The application is built with **Streamlit** for an intuitive user interface and includes dynamic visualizations using **Plotly**.

### Features
- Display of a player's historical performance (points, rebounds, assists, etc.) over the last games.
- Simulation of performance for the 2025-26 season based on moving averages and a predictive model.
- Interactive data visualizations (actual vs. predicted).
- Display of the player's current team (requires manual updates).
- Customizable settings via a sidebar (number of games to analyze).


### Prerequisites
- Python 3.8 or higher
- Required libraries:
  - `streamlit`
  - `nba_api`
  - `plotly`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Usage
1. Run the application:
streamlit run NBA_Analytics.py
2. In the sidebar, select a player (e.g., Mo Bamba, Shai Gilgeous-Alexander) and adjust the number of games to analyze.
3. Explore historical performance, mean squared errors (MSE), and the 2025-26 season simulation.
4. Enable optional graphs for a detailed analysis.

Contribution
-Open an issue to report bugs or suggest improvements.
-Submit pull requests for additions (e.g., new metrics, automation of current teams).


Acknowledgments
-Thanks to the NBA API for the data.
-Inspired by Streamlit tutorials and scikit-learn resources.


Author
Trésor Déo-Gratias ZINGUEDE - tresordeozinguede@gmail.com








