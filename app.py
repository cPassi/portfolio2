from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    data = {
        'name': 'Pascal Potthoff',
        'age': 21,
        'location': 'Schloß Holte-Stukenbrock',
        'bio': (
            'Leidenschaftlicher Fußballer mit großem Interesse an Cyber Security und AI. '
            'Ich studiere Wirtschaftsinformatik (B.Sc.) dual an der FHDW Paderborn berufsbegleitend '
            'bei Bertelsmann SE & Co. KGaA.'
        ),
        'education': [
            # {
            #     'title': 'Gymnasium Schloß Holte-Stukenbrock',
            #     'period': '–'
            # },
            {
                'title': 'Bachelor of Science Wirtschaftsinformatik (Cyber Security)',
                'institution': 'FHDW Paderborn (dual)',
                'period': 'Aug 2022 – Feb 2026',
                'thesis': 'Bachelor Thesis: Einführung von Threat Hunting in ein SOC: Herausforderungen, Potenziale und Erfolgsfaktoren'
            },
            {
                'title': 'Ausbildung Fachinformatiker Fachrichtung Anwendungsentwicklung',
                'institution': 'während des Studiums',
                'period': 'bis Juni 2024',
                
            }
        ],
        'internships': [
            {
                'department': 'eHealth (eGK und KIM)',
                'period': 'Aug 2022 – Juni 2023'
            },
            {
                 'department': 'SAP FI/CO – Betreuung & Beratung',
                 'period': 'Juli 2023 – heute',
                 
             
            },
            {
                'department': 'Cyber Security inkl. Bachelor Thesis',
                'period': 'Aug 2025 – heute'
            }
        ],
        'project': {
            'title': 'Kostenmonitor in Python',
            'description': (
                'Automatische Erstellung von Grafiken auf Basis von Excel-Tabellen '
                'und Export als PowerPoint'
            )
        },
        'skills': [
            'Python', 'ABAP', 'Java', 'C',
            'SAP FI Buchungen', 'SQL', 'Datenanalyse'
        ],
        'strengths': [
            'Analytisches Denken',
            'Teamarbeit'
        ],
        'hobbies': [
            'Fußball',
            'TryHackMe',
            'Backen'
        ]
    }
    return render_template('about.html', **data)

@app.route('/projects')
def projects():
    project_list = [
        {
            'title': 'Kostenmonitor für SAP BTP-Services',
            'summary': (
                'Entwicklung eines automatisierten Tools in Python zur Auswertung '
                'von Excel-Kostendaten, Visualisierung in Diagrammen und Export '
                'als PowerPoint-Präsentation.'
            ),
            'technologies': ['Python', 'pandas', 'matplotlib', 'openpyxl', 'python-pptx']
        },
        {
            'title': 'Dortmund Match Predictor',
            'summary': (
                'Machine Learning Modell zur Vorhersage von Torerwartungen für '
                'Borussia Dortmund Fußballspiele. Nutzt Linear Regression mit '
                'Features wie Elo-Ratings, aktuelle Form und Head-to-Head Statistiken.'
            ),
            'technologies': ['Python', 'Flask', 'Scikit-Learn', 'Machine Learning', 'Pandas', 'HTML/CSS/JS'],
            'link': '/predictor'
        }
    ]
    return render_template('projects.html', projects=project_list)


# ============================================================
# DORTMUND PREDICTOR - SETUP BEIM APP START
# ============================================================

csv_path = 'C:/Users/Pascal Potthoff/footballpredictor/data/processed/dortmund_with_h2h.csv'
dortmund_df = None
model = None
opponents = []

if os.path.exists(csv_path):
    try:
        dortmund_df = pd.read_csv(csv_path)
        dortmund_df['MatchDate'] = pd.to_datetime(dortmund_df['MatchDate'])
        
        # Filtere nur Teams der letzten 3 Jahre
        cutoff_date = dortmund_df['MatchDate'].max() - timedelta(days=3*365)
        valid_teams_mask = dortmund_df.groupby('Opponent')['MatchDate'].transform('max') >= cutoff_date
        dortmund_df = dortmund_df[valid_teams_mask].copy()
        
        # Trainiere Modell
        feature_columns = [
            'IsHome', 'DortmundElo', 'OpponentElo', 'DortmundForm5', 'OpponentForm5',
            'H2H_Games', 'H2H_DortmundWins', 'H2H_Draws', 'H2H_DortmundLosses',
            'H2H_AvgGoalsFor', 'H2H_AvgGoalsAgainst'
        ]
        
        X = dortmund_df[feature_columns]
        y = dortmund_df['DortmundGoals']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        opponents = sorted(dortmund_df['Opponent'].unique().tolist())
        print("✅ Dortmund Predictor geladen")
    except Exception as e:
        print(f"⚠️  Predictor konnte nicht geladen werden: {e}")

# ============================================================
# PREDICTOR ROUTES
# ============================================================

@app.route('/predictor')
def predictor():
    """Predictor Page"""
    if model is None:
        return "Error: Predictor data not loaded", 500
    return render_template('predictor.html', opponents=opponents)

@app.route('/predict', methods=['POST'])
def predict():
    """API für Vorhersagen"""
    try:
        if model is None or dortmund_df is None:
            return jsonify({'success': False, 'error': 'Predictor not loaded'}), 500
        
        data = request.json
        opponent = data.get('opponent')
        is_home = int(data.get('is_home'))
        
        opponent_data = dortmund_df[dortmund_df['Opponent'] == opponent].tail(1)
        
        if len(opponent_data) == 0:
            return jsonify({'success': False, 'error': f'Gegner nicht gefunden'}), 400
        
        input_features = [
            is_home,
            opponent_data['DortmundElo'].values[0],
            opponent_data['OpponentElo'].values[0],
            opponent_data['DortmundForm5'].values[0],
            opponent_data['OpponentForm5'].values[0],
            opponent_data['H2H_Games'].values[0],
            opponent_data['H2H_DortmundWins'].values[0],
            opponent_data['H2H_Draws'].values[0],
            opponent_data['H2H_DortmundLosses'].values[0],
            opponent_data['H2H_AvgGoalsFor'].values[0],
            opponent_data['H2H_AvgGoalsAgainst'].values[0],
        ]
        
        prediction = model.predict([input_features])[0]
        
        return jsonify({
            'success': True,
            'prediction': round(max(0, prediction), 2),
            'opponent': opponent,
            'location': 'Heimspiel' if is_home == 1 else 'Auswärtsspiel',
            'h2h_games': int(opponent_data['H2H_Games'].values[0]),
            'h2h_wins': int(opponent_data['H2H_DortmundWins'].values[0]),
            'h2h_draws': int(opponent_data['H2H_Draws'].values[0]),
            'h2h_losses': int(opponent_data['H2H_DortmundLosses'].values[0]),
            'dortmund_elo': round(opponent_data['DortmundElo'].values[0], 0),
            'opponent_elo': round(opponent_data['OpponentElo'].values[0], 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run()