# Datasets

`OutlierDetectionData` provides access to collections of outlier detection datasets. The following collections are currently supported:

- [ODDS](http://odds.cs.stonybrook.edu/), Outlier Detection DataSets, Shebuti Rayana, 2016
- [ELKI](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/), On the Evaluation of Unsupervised Outlier Detection, Campos et al., 2016
- [TSAD](https://timeseriesclassification.com/), The UCR Time Series Archive, Dau et al., 2018

For the TSAD collection, the class with the least members is chosen as the anomaly class and all other
classes are defined as normal. If there are multiple classes, the lexically first class is chosen.

The following methods are defined for all collections.

## `list`

```@docs
OutlierDetectionData.list
```

## `load`

```@docs
OutlierDetectionData.load
```
