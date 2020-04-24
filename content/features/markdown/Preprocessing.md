# Recommended Preprocessing

There are currently no agreed upon conventions for preprocessing naturalistic neuroimaging data.

In our work, we perform standard realignment and spatial normalization. We also perform basic denosing by removing global and multivariate spikes, average CSF activity, linear/quadratic trends, and 24 motion covariates (e.g., 6 centered realignment, their squares, derivative, and squared derivatives).
