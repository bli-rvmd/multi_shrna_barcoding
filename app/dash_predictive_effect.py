import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# Load the predictive effect dataset
df_predictive_effect_path = "~/Documents/Projects/Multi_shRNA_screening_AP009/data/predictive_effect.csv"  # Ensure correct file path
df_predictive_effect = pd.read_csv(df_predictive_effect_path)

# Sort timepoints in order
timepoint_order = ["T7", "T10", "T13"]
df_predictive_effect["Timepoint"] = pd.Categorical(df_predictive_effect["Timepoint"], categories=timepoint_order, ordered=True)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Predictive Effect Dashboard"),
    
    # Dropdown input for selecting a target gene
    html.Label("Select Target Gene:"),
    dcc.Dropdown(
        id='target-gene-dropdown',
        options=[{'label': gene, 'value': gene} for gene in sorted(df_predictive_effect["Target_Gene"].unique())],
        value=sorted(df_predictive_effect["Target_Gene"].unique())[0],
        clearable=False
    ),
    
    # Line plot output
    dcc.Graph(id='predictive-effect-lineplot')
])

@app.callback(
    Output('predictive-effect-lineplot', 'figure'),
    Input('target-gene-dropdown', 'value')
)
def update_plot(selected_gene):
    # Filter data for selected target gene
    df_filtered = df_predictive_effect[df_predictive_effect["Target_Gene"] == selected_gene]
    
    # Generate line plot
    fig = px.line(
        df_filtered, x="Timepoint", y="Predictive_Effect", color="shRNA_ID",
        facet_col="Dosage", markers=True,
        title=f"Predictive Effect for {selected_gene}",
        labels={"Predictive_Effect": "Predictive Effect (RTN_A / RTN_B)", "Timepoint": "Timepoint"}
    )
    
    fig.update_xaxes(categoryorder='array', categoryarray=timepoint_order)
    fig.update_layout(legend_title_text='shRNA_ID', height=600)
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
