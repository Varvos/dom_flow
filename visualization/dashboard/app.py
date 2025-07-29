import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from src.flows import SimpleAveraging, WeightedAveraging, StochasticMatrixFlow
from visualization.utils import circle, flower, star, rectangle, spiral

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Domain Evolution Flow Visualization", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Shape Parameters"),
            html.Label("Shape Type:"),
            dcc.Dropdown(
                id='shape-type',
                options=[
                    {'label': 'Circle', 'value': 'circle'},
                    {'label': 'Flower', 'value': 'flower'},
                    {'label': 'Star', 'value': 'star'},
                    {'label': 'Rectangle', 'value': 'rectangle'},
                    {'label': 'Spiral', 'value': 'spiral'}
                ],
                value='circle'
            ),
            
            html.Label("Number of Points:"),
            dcc.Slider(id='n-points', min=10, max=200, step=10, value=50, 
                      marks={i: str(i) for i in range(10, 201, 40)}),
            
            html.Label("Scale X:"),
            dcc.Slider(id='scale-x', min=0.1, max=3.0, step=0.1, value=1.0,
                      marks={i/10: f'{i/10:.1f}' for i in range(1, 31, 5)}),
            
            html.Label("Scale Y:"),
            dcc.Slider(id='scale-y', min=0.1, max=3.0, step=0.1, value=1.0,
                      marks={i/10: f'{i/10:.1f}' for i in range(1, 31, 5)}),
            
            # Shape-specific parameters
            html.Div(id='shape-specific-params'),
            
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            html.H3("Flow Parameters"),
            html.Label("Flow Type:"),
            dcc.Dropdown(
                id='flow-type',
                options=[
                    {'label': 'Simple Averaging', 'value': 'simple'},
                    {'label': 'Weighted Averaging', 'value': 'weighted'},
                    {'label': 'Custom Stochastic Matrix', 'value': 'custom'}
                ],
                value='simple'
            ),
            
            html.Div(id='flow-specific-params'),
            
            html.Label("Evolution Steps:"),
            dcc.Slider(id='steps', min=0, max=100, step=1, value=10,
                      marks={i: str(i) for i in range(0, 101, 20)}),
            
            html.Br(),
            html.Button('Run Evolution', id='run-button', n_clicks=0,
                       style={'fontSize': '16px', 'padding': '10px 20px'}),
            
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            html.H3("Animation Controls"),
            html.Label("Animation Speed (ms):"),
            dcc.Slider(id='animation-speed', min=100, max=2000, step=100, value=500,
                      marks={i: f'{i}ms' for i in range(100, 2001, 400)}),
            
            html.Br(),
            html.Button('Play Animation', id='play-button', n_clicks=0,
                       style={'fontSize': '16px', 'padding': '10px 20px'}),
            
            dcc.Interval(id='interval-component', interval=500, n_intervals=0, disabled=True),
            
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    ]),
    
    html.Div([
        dcc.Graph(id='evolution-plot', style={'height': '600px'})
    ], style={'padding': '20px'}),
    
    # Store components for data
    dcc.Store(id='trajectory-data'),
    dcc.Store(id='animation-state', data={'playing': False, 'current_step': 0})
])


@app.callback(
    Output('shape-specific-params', 'children'),
    Input('shape-type', 'value')
)
def update_shape_params(shape_type):
    if shape_type == 'flower':
        return [
            html.Label("Number of Petals:"),
            dcc.Slider(id='petals', min=3, max=12, step=1, value=5,
                      marks={i: str(i) for i in range(3, 13, 2)})
        ]
    elif shape_type == 'star':
        return [
            html.Label("Number of Spikes:"),
            dcc.Slider(id='spikes', min=3, max=12, step=1, value=5,
                      marks={i: str(i) for i in range(3, 13, 2)})
        ]
    elif shape_type == 'rectangle':
        return [
            html.Label("Width:"),
            dcc.Slider(id='width', min=0.5, max=4.0, step=0.1, value=2.0,
                      marks={i/2: f'{i/2:.1f}' for i in range(1, 9, 2)}),
            html.Label("Height:"),
            dcc.Slider(id='height', min=0.5, max=4.0, step=0.1, value=1.0,
                      marks={i/2: f'{i/2:.1f}' for i in range(1, 9, 2)})
        ]
    elif shape_type == 'spiral':
        return [
            html.Label("Number of Turns:"),
            dcc.Slider(id='turns', min=1, max=8, step=0.5, value=3.0,
                      marks={i: str(i) for i in range(1, 9, 2)})
        ]
    return []


@app.callback(
    Output('flow-specific-params', 'children'),
    Input('flow-type', 'value')
)
def update_flow_params(flow_type):
    if flow_type == 'weighted':
        return [
            html.Label("Neighbor Weight:"),
            dcc.Slider(id='neighbor-weight', min=0.1, max=0.9, step=0.1, value=0.5,
                      marks={i/10: f'{i/10:.1f}' for i in range(1, 10, 2)}),
            html.P("Self weight will be 1 - neighbor_weight")
        ]
    elif flow_type == 'custom':
        return [
            html.Label("Custom weights (comma-separated):"),
            dcc.Input(id='custom-weights', type='text', value='0.3,0.4,0.2,0.1',
                     placeholder='e.g., 0.3,0.4,0.2,0.1')
        ]
    return []


def generate_shape(shape_type, n_points, scale_x, scale_y, **kwargs):
    """Generate shape points based on parameters."""
    if shape_type == 'circle':
        return circle(n_points, scale_x=scale_x, scale_y=scale_y)
    elif shape_type == 'flower':
        petals = kwargs.get('petals', 5)
        return flower(n_points, petals=petals, scale_x=scale_x, scale_y=scale_y)
    elif shape_type == 'star':
        spikes = kwargs.get('spikes', 5)
        return star(n_points, spikes=spikes, scale_x=scale_x, scale_y=scale_y)
    elif shape_type == 'rectangle':
        width = kwargs.get('width', 2.0)
        height = kwargs.get('height', 1.0)
        return rectangle(n_points, width=width, height=height, scale_x=scale_x, scale_y=scale_y)
    elif shape_type == 'spiral':
        turns = kwargs.get('turns', 3.0)
        return spiral(n_points, turns=turns, scale_x=scale_x, scale_y=scale_y)


def create_flow(flow_type, domain_size, **kwargs):
    """Create flow object based on parameters."""
    if flow_type == 'simple':
        return SimpleAveraging(domain_size)
    elif flow_type == 'weighted':
        neighbor_weight = kwargs.get('neighbor_weight', 0.5)
        self_weight = 1 - neighbor_weight
        weights = np.zeros(domain_size)
        weights[0] = self_weight
        weights[1] = neighbor_weight
        return WeightedAveraging(weights)
    elif flow_type == 'custom':
        weights_str = kwargs.get('custom_weights', '0.5,0.5')
        try:
            weights = np.array([float(w.strip()) for w in weights_str.split(',')])
            # Pad or truncate to domain_size
            if len(weights) < domain_size:
                padded = np.zeros(domain_size)
                padded[:len(weights)] = weights
                weights = padded
            else:
                weights = weights[:domain_size]
            # Normalize
            weights = weights / weights.sum()
            return WeightedAveraging(weights)
        except:
            # Fallback to simple averaging
            return SimpleAveraging(domain_size)


@app.callback(
    Output('trajectory-data', 'data'),
    [Input('run-button', 'n_clicks')],
    [State('shape-type', 'value'),
     State('n-points', 'value'),
     State('scale-x', 'value'),
     State('scale-y', 'value'),
     State('flow-type', 'value'),
     State('steps', 'value')] +
    [State('petals', 'value')] +
    [State('spikes', 'value')] +
    [State('width', 'value')] +
    [State('height', 'value')] +
    [State('turns', 'value')] +
    [State('neighbor-weight', 'value')] +
    [State('custom-weights', 'value')]
)
def run_evolution(n_clicks, shape_type, n_points, scale_x, scale_y, flow_type, steps,
                 petals, spikes, width, height, turns, neighbor_weight, custom_weights):
    if n_clicks == 0:
        return None
    
    # Generate initial shape
    shape_kwargs = {}
    if shape_type == 'flower' and petals is not None:
        shape_kwargs['petals'] = petals
    elif shape_type == 'star' and spikes is not None:
        shape_kwargs['spikes'] = spikes
    elif shape_type == 'rectangle':
        if width is not None:
            shape_kwargs['width'] = width
        if height is not None:
            shape_kwargs['height'] = height
    elif shape_type == 'spiral' and turns is not None:
        shape_kwargs['turns'] = turns
    
    points_2d = generate_shape(shape_type, n_points, scale_x, scale_y, **shape_kwargs)
    
    # Flatten to 1D for flow evolution
    z_init = points_2d.flatten()  # [x1, y1, x2, y2, ...]
    domain_size = len(z_init)
    
    # Create flow
    flow_kwargs = {}
    if flow_type == 'weighted' and neighbor_weight is not None:
        flow_kwargs['neighbor_weight'] = neighbor_weight
    elif flow_type == 'custom' and custom_weights is not None:
        flow_kwargs['custom_weights'] = custom_weights
    
    flow = create_flow(flow_type, domain_size, **flow_kwargs)
    
    # Run evolution
    trajectory = flow.evolve(z_init, steps)
    
    # Convert back to 2D points for each step
    trajectory_2d = []
    for step in range(trajectory.shape[0]):
        step_data = trajectory[step].reshape(-1, 2)
        trajectory_2d.append({
            'x': step_data[:, 0].tolist(),
            'y': step_data[:, 1].tolist(),
            'step': step
        })
    
    return trajectory_2d


@app.callback(
    [Output('evolution-plot', 'figure'),
     Output('interval-component', 'disabled'),
     Output('animation-state', 'data')],
    [Input('trajectory-data', 'data'),
     Input('play-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('animation-speed', 'value'),
     State('animation-state', 'data')]
)
def update_plot(trajectory_data, play_clicks, n_intervals, animation_speed, animation_state):
    ctx = dash.callback_context
    
    if trajectory_data is None:
        return go.Figure(), True, animation_state
    
    # Handle play button
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'play-button.n_clicks':
        animation_state['playing'] = not animation_state.get('playing', False)
        animation_state['current_step'] = 0
        return dash.no_update, not animation_state['playing'], animation_state
    
    # Handle animation interval
    if animation_state.get('playing', False) and ctx.triggered and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        current_step = animation_state.get('current_step', 0)
        if current_step < len(trajectory_data) - 1:
            animation_state['current_step'] = current_step + 1
        else:
            animation_state['playing'] = False
            return dash.no_update, True, animation_state
    
    # Create figure
    fig = go.Figure()
    
    if animation_state.get('playing', False):
        # Show animation up to current step
        current_step = animation_state.get('current_step', 0)
        steps_to_show = min(current_step + 1, len(trajectory_data))
    else:
        # Show all steps
        steps_to_show = len(trajectory_data)
    
    # Add traces for each step
    colors = np.linspace(0, 1, len(trajectory_data))
    for i in range(steps_to_show):
        step_data = trajectory_data[i]
        
        fig.add_trace(go.Scatter(
            x=step_data['x'] + [step_data['x'][0]],  # Close the curve
            y=step_data['y'] + [step_data['y'][0]],
            mode='lines+markers',
            name=f'Step {i}',
            line=dict(color=f'rgba({int(255*colors[i])}, {int(255*(1-colors[i]))}, 100, 0.8)'),
            marker=dict(size=4),
            opacity=0.7 if i < steps_to_show - 1 else 1.0
        ))
    
    fig.update_layout(
        title=f"Domain Evolution - Step {steps_to_show-1}/{len(trajectory_data)-1}",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        template="plotly_white"
    )
    
    return fig, not animation_state.get('playing', False), animation_state


if __name__ == '__main__':
    app.run_server(debug=True)