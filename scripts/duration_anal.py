import json
from collections import Counter
import plotly.graph_objects as go

# Cargar el JSON
with open('chart_analysis_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

duration = []

#data["songs"][0]["sections"]["ExpertSingle"]
#import pdb;pdb.set_trace()
for song in data["songs"]:
    sections = list(song["sections"].keys())
    notes_sections = [section for section in sections if section.lower().startswith(("expert", "hard", "medium", "easy"))]
    for section in notes_sections:
        for k,v in song["sections"][section].items():
            if len(v) == 3:
                duration.append(v[2]/192)  # Asumiendo que el tercer elemento es la duración
print(duration)

# Calcular el histograma
histograma = Counter(duration)
print(histograma)
# Preparar datos para plotly
x = list(histograma.keys())
y = list(histograma.values())

fig = go.Figure(data=[go.Bar(x=x, y=y)])
fig.update_layout(title='Histograma De Duraciones de Notas',
                  xaxis_title='Valor',
                  yaxis_title='Frecuencia')
fig.show()