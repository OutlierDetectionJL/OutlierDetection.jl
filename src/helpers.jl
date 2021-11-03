"""
    outlier_fraction(y)

Determine the fraction of outliers in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    outlier_fraction::Float64
The fraction of outliers.
"""
outlier_fraction(y::Labels) = sum(y .== OutlierDetection.CLASS_OUTLIER) / length(y)

"""
    n_normal(y)
Determine the amount of normal data points in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    n_normal::Int64
The amount of normal data points.
"""
n_normal(y::Labels) = sum(y .== OutlierDetection.CLASS_NORMAL)

"""
    n_outlier(y)
Determine the amount of outlier data points in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    outliers::Int64
The amount of outlier data points.
"""
n_outlier(y::Labels) = sum(y .== OutlierDetection.CLASS_OUTLIER)
