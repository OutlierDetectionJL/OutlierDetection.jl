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
    normal_count(y)
Determine the count of normals in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    normal_count::Int64
The count of normals.
"""
normal_count(y::Labels) = sum(y .== OutlierDetection.CLASS_NORMAL)

"""
    outlier_count(y)
Determine the count of outliers in a given vector.

Parameters
----------
    y::Labels
An array containing "normal" and "outlier" classes.

Returns
----------
    outlier_count::Int64
The count of outliers.
"""
outlier_count(y::Labels) = sum(y .== OutlierDetection.CLASS_OUTLIER)
