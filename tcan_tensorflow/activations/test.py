'''
Plot the entmax activations as in Figure 3 in Peters, B., Niculae, V., & Martins, A. F. (2019).
Sparse sequence-to-sequence models. https://arxiv.org/abs/1905.05702.
'''

import numpy as np
import plotly.graph_objects as go

from tcan_tensorflow.activations.activations import entmax

# Define the values of the sparsity parameter and the corresponding legend items.
alphas = {
    'values': [1, 1.25, 1.5, 2, 4],
    'names': [r'$\alpha=1\text{ (softmax)}$', r'$\alpha=1.25$', r'$\alpha=1.5$', r'$\alpha=2\text{ (sparsemax)}$', r'$\alpha=4$'],
    'colors': ['#57606A', '#CF222E', '#000000', '#0550AE', '#8250DF'],
    'dash': ['dot', None, 'dash', None, None]
}

# Calculate the entmax activations between -3 and +3 for the different values of the sparsity parameter.
t = np.linspace(-3, 3, 100)
z = np.hstack([t.reshape(- 1, 1), np.zeros((100, 1))])
p = [entmax(z, alpha)[:, 0].numpy() for alpha in alphas['values']]

# Plot the entmax activations.
layout = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(t=10, b=10, l=40, r=10),
    font=dict(
        color='#000000',
        size=10,
    ),
    legend=dict(
        x=0,
        y=1,
        font=dict(
            color='#000000',
        ),
    ),
    xaxis=dict(
        title=r'$t$',
        range=[-3, 3],
        color='#000000',
        tickfont=dict(
            color='#3a3a3a',
        ),
        linecolor='#d9d9d9',
        mirror=True,
        showgrid=False,
    ),
    yaxis=dict(
        title=r'$\alpha\text{-entmax}([t, 0])$',
        color='#000000',
        tickfont=dict(
            color='#3a3a3a',
        ),
        linecolor='#d9d9d9',
        mirror=True,
        showgrid=False,
        zeroline=False,
    ),
)

data = []

for i in range(len(alphas['values'])):

    data.append(
        go.Scatter(
            x=t,
            y=p[i],
            mode='lines',
            name=alphas['names'][i],
            line=dict(
                color=alphas['colors'][i],
                dash=alphas['dash'][i],
                width=1,
            )
        )
    )

fig = go.Figure(data=data, layout=layout)

# Save the plot.
fig.write_image('activations.png', width=650, height=400)
