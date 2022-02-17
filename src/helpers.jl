"""
    normal_fraction(y)

Determine the fraction of normals in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    outlier_fraction::Float64
The fraction of normals.
"""
normal_fraction(y::Labels) = sum(y .== OutlierDetection.CLASS_NORMAL) / length(y)

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
Determine the count of normals in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    n_normal::Int64
The count of normals.
"""
n_normal(y::Labels) = sum(y .== OutlierDetection.CLASS_NORMAL)

"""
    n_outlier(y)
Determine the count of outliers in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    outliers::Int64
The count of outliers.
"""
n_outlier(y::Labels) = sum(y .== OutlierDetection.CLASS_OUTLIER)
