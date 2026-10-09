"""Projected empirical optimal transport on seasonal level and transition measures."""
import numpy as np

def monthly_flow_quantiles(flow, months, n_quantiles=32):
    """Monthly quantiles of log(1 + Q/mean(Q)), with equal month weights.

    Quantiles use linear interpolation of the empirical sample. The resulting
    Euclidean distance approximates the mean of twelve conditional W2 squares.
    Zero flows are retained; missing and negative flows are excluded.
    """
    flow=np.asarray(flow,float);months=np.asarray(months)
    if flow.ndim!=1 or months.shape!=flow.shape or n_quantiles<1:
        raise ValueError('Incompatible flow, month, or quantile arguments')
    valid=np.isfinite(flow)&(flow>=0)
    if not valid.any() or flow[valid].mean()<=0:
        raise ValueError('Positive mean flow is required')
    x=np.log1p(flow[valid]/flow[valid].mean());m=months[valid]
    probabilities=(np.arange(n_quantiles)+.5)/n_quantiles
    if any(not np.any(m==month) for month in range(1,13)):
        raise ValueError('Every calendar month must have observations')
    return np.stack([np.quantile(x[m==month],probabilities) for month in range(1,13)])

def projected_quantiles(points,weights,directions,n_quantiles=128):
    points=np.asarray(points,float);weights=np.asarray(weights,float)
    directions=np.asarray(directions,float)
    if points.ndim!=2 or weights.shape!=(len(points),):
        raise ValueError('Incompatible support and weights')
    if np.any(weights<0) or not weights.sum()>0:
        raise ValueError('Weights must define a nonnegative measure')
    weights=weights/weights.sum()
    projections=points@directions.T
    probabilities=(np.arange(n_quantiles)+.5)/n_quantiles
    out=np.empty((len(directions),n_quantiles))
    for j in range(len(directions)):
        order=np.argsort(projections[:,j],kind='stable')
        cumulative=np.cumsum(weights[order]);cumulative[-1]=1.
        indices=np.searchsorted(cumulative,probabilities,side='left')
        out[j]=projections[order[indices],j]
    return out
