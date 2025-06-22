import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from rf_utils import load_data

# Load model and splits
model = joblib.load("best_model.pkl")
X, _, feature_names = load_data("synthetic_data.csv")
X_train, X_test, y_train, y_test = joblib.load("data/splits.pkl")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ROC and PR metrics
fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importances
feat_imp = model.named_steps["classifier"].feature_importances_

# CV scores
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=89)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

# Learning curve
train_sizes, tr_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# ROC Plot
roc_figure = go.Figure()
roc_figure.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode="lines",
    showlegend=False,
    line=dict(color="blue")
))
roc_figure.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode="markers",
    marker=dict(size=10, color="blue"),
    customdata=[[a, b, c] for a, b, c in zip(roc_thresh, tpr, fpr)],
    hovertemplate="Threshold: %{customdata[0]:.2f}<br>TPR: %{customdata[1]:.2f}<br>FPR: %{customdata[2]:.2f}<extra></extra>",
    showlegend=False
))
roc_figure.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

# PR Plot
pr_figure = go.Figure()
pr_figure.add_trace(go.Scatter(
    x=recall,
    y=precision,
    mode="lines",
    showlegend=False,
    line=dict(color="green")
))
pr_figure.add_trace(go.Scatter(
    x=recall[:-1],
    y=precision[:-1],
    mode="markers",
    marker=dict(size=10, color="green"),
    customdata=[[a, b, c] for a, b, c in zip(pr_thresh, precision[:-1], recall[:-1])],
    hovertemplate="Threshold: %{customdata[0]:.2f}<br>Precision: %{customdata[1]:.2f}<br>Recall: %{customdata[2]:.2f}<extra></extra>",
    showlegend=False
))
pr_figure.update_layout(xaxis_title="Recall", yaxis_title="Precision")

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Model Performance Dashboard"

# App layout
app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "20px"}, children=[
    html.H1("Model Performance Dashboard"),

    html.Div([
        html.H2("ROC Curve"),
        html.P("Click any point on the plot to see interpretation below.", style={"fontStyle": "italic", "color": "#555"}),
        dcc.Graph(id="roc-plot", figure=roc_figure),
        html.Div(id="roc-summary", style={"marginTop": "10px", "fontStyle": "italic", "color": "#333"}),
        html.P("This plot shows the tradeoff between true positive rate and false positive rate at different classification thresholds.")
    ]),

    html.Div([
        html.H2("Precision-Recall Curve"),
        html.P("Click any point on the plot to see interpretation below.", style={"fontStyle": "italic", "color": "#555"}),
        dcc.Graph(id="pr-plot", figure=pr_figure),
        html.Div(id="pr-summary", style={"marginTop": "10px", "fontStyle": "italic", "color": "#333"}),
        html.P("This plot shows how precision and recall vary with the threshold, useful for imbalanced datasets.")
    ]),

    html.Div([
        html.H2("Confusion Matrix"),
        dcc.Graph(
            figure=go.Figure(go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues"
            ))
        ),
        html.P("The confusion matrix shows how many true positives, false positives, true negatives, and false negatives the model predicted.")
    ]),

    html.Div([
        html.H2("Learning Curve (F1 Score)"),
        dcc.Graph(
            figure=go.Figure([
                go.Scatter(x=train_sizes, y=np.mean(tr_scores, axis=1), mode="lines+markers", name="Train"),
                go.Scatter(x=train_sizes, y=np.mean(val_scores, axis=1), mode="lines+markers", name="Validation")
            ]).update_layout(xaxis_title="Training Examples", yaxis_title="F1 Score")
        ),
        html.P("This curve shows how training and validation performance change as we increase the training dataset size.")
    ]),

    html.Div([
        html.H2("3-Fold CV F1 Scores"),
        dcc.Graph(
            figure=go.Figure(go.Box(y=cv_scores, boxpoints="all")).update_layout(yaxis_title="F1 Score")
        ),
        html.P("Shows the F1 scores across 3 cross-validation splits to measure generalization.")
    ]),

    html.Div([
        html.H2("Feature Importances"),
        dcc.Graph(
            figure=go.Figure(go.Bar(
                x=feature_names,
                y=feat_imp
            )).update_layout(xaxis_title="Feature", yaxis_title="Importance")
        ),
        html.P("Ranks which input features the model relied on most to make its predictions.")
    ]),

    html.H2("Glossary of Key Terms", style={"marginTop": "40px"}),
    html.Ul([
        html.Li([html.B("Precision"), ": The percentage of predicted positives that are actually positive."]),
        html.Li([html.B("Recall"), ": The percentage of actual positives that were correctly identified."]),
        html.Li([html.B("F1 Score"), ": Harmonic mean of precision and recall. Balances both metrics."]),
        html.Li([html.B("Learning Rate"), ": Controls how fast the model updates. Smaller is slower but safer."]),
        html.Li([html.B("Threshold"), ": The cutoff probability above which the model classifies a prediction as positive."]),
        html.Li([html.B("ROC Curve"), ": Visualizes tradeoff between true and false positive rates."]),
        html.Li([html.B("Confusion Matrix"), ": Table showing correct vs incorrect predictions for each class."]),
        html.Li([html.B("Cross-Validation"), ": Technique to evaluate how well the model generalizes."]),
    ], style={"lineHeight": "1.8"})
])

# Callbacks
@app.callback(Output("roc-summary", "children"), Input("roc-plot", "clickData"))
def update_roc_summary(clickData):
    print("ROC ClickData:", clickData)
    if clickData and "customdata" in clickData["points"][0]:
        thresh, tpr_, fpr_ = clickData["points"][0]["customdata"]
        return f"At threshold {thresh:.2f}, the model captures {tpr_ * 100:.1f}% of true positives and allows {fpr_ * 100:.1f}% false positives."
    return "Click a point to see interpretation."

@app.callback(Output("pr-summary", "children"), Input("pr-plot", "clickData"))
def update_pr_summary(clickData):
    print("PR ClickData:", clickData)
    if clickData and "customdata" in clickData["points"][0]:
        thresh, prec_, rec_ = clickData["points"][0]["customdata"]
        return (f"At threshold {thresh:.2f}, precision is {prec_ * 100:.1f}%, and recall is {rec_ * 100:.1f}%.")
    return "Click a point to see interpretation."

if __name__ == "__main__":
    app.run(debug=True)
